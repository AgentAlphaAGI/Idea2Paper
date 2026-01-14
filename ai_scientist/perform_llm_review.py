import os
import json
import numpy as np
from pypdf import PdfReader
import pymupdf
import pymupdf4llm
from ai_scientist.llm import (
    get_response_from_llm,
    get_batch_responses_from_llm,
    extract_json_between_markers,
)
from ai_scientist.prompts.loader import load_prompt, render_prompt

reviewer_system_prompt_base = load_prompt(
    "perform_llm_review.reviewer_system_prompt_base"
)
reviewer_system_prompt_neg = (
    reviewer_system_prompt_base
    + load_prompt("perform_llm_review.reviewer_system_prompt_neg_suffix")
)
reviewer_system_prompt_pos = (
    reviewer_system_prompt_base
    + load_prompt("perform_llm_review.reviewer_system_prompt_pos_suffix")
)

template_instructions = load_prompt("perform_llm_review.template_instructions")

neurips_form = load_prompt("perform_llm_review.neurips_form_base") + template_instructions


def perform_review(
    text,
    model,
    client,
    num_reflections=1,
    num_fs_examples=1,
    num_reviews_ensemble=1,
    temperature=0.75,
    msg_history=None,
    return_msg_history=False,
    reviewer_system_prompt=reviewer_system_prompt_neg,
    review_instruction_form=neurips_form,
):
    if num_fs_examples > 0:
        fs_prompt = get_review_fewshot_examples(num_fs_examples)
        base_prompt = review_instruction_form + fs_prompt
    else:
        base_prompt = review_instruction_form

    base_prompt += render_prompt(
        "perform_llm_review.review_paper_append", text=text
    )

    if num_reviews_ensemble > 1:
        llm_reviews, msg_histories = get_batch_responses_from_llm(
            base_prompt,
            model=model,
            client=client,
            system_message=reviewer_system_prompt,
            print_debug=False,
            msg_history=msg_history,
            temperature=0.75,
            n_responses=num_reviews_ensemble,
        )
        parsed_reviews = []
        for idx, rev in enumerate(llm_reviews):
            try:
                parsed_reviews.append(extract_json_between_markers(rev))
            except Exception as e:
                print(f"Ensemble review {idx} failed: {e}")
        parsed_reviews = [r for r in parsed_reviews if r is not None]
        review = get_meta_review(model, client, temperature, parsed_reviews)
        if review is None:
            review = parsed_reviews[0]
        for score, limits in [
            ("Originality", (1, 4)),
            ("Quality", (1, 4)),
            ("Clarity", (1, 4)),
            ("Significance", (1, 4)),
            ("Soundness", (1, 4)),
            ("Presentation", (1, 4)),
            ("Contribution", (1, 4)),
            ("Overall", (1, 10)),
            ("Confidence", (1, 5)),
        ]:
            scores = []
            for r in parsed_reviews:
                if score in r and limits[0] <= r[score] <= limits[1]:
                    scores.append(r[score])
            if scores:
                review[score] = int(round(np.mean(scores)))
        msg_history = msg_histories[0][:-1]
        msg_history += [
            {
                "role": "assistant",
                "content": render_prompt(
                    "perform_llm_review.ensemble_assistant_message",
                    num_reviews_ensemble=num_reviews_ensemble,
                    review_json=json.dumps(review),
                ),
            }
        ]
    else:
        llm_review, msg_history = get_response_from_llm(
            base_prompt,
            model=model,
            client=client,
            system_message=reviewer_system_prompt,
            print_debug=False,
            msg_history=msg_history,
            temperature=temperature,
        )
        review = extract_json_between_markers(llm_review)

    if num_reflections > 1:
        for j in range(num_reflections - 1):
            text, msg_history = get_response_from_llm(
                reviewer_reflection_prompt,
                client=client,
                model=model,
                system_message=reviewer_system_prompt,
                msg_history=msg_history,
                temperature=temperature,
            )
            review = extract_json_between_markers(text)
            assert review is not None, "Failed to extract JSON from LLM output"
            if "I am done" in text:
                break

    if return_msg_history:
        return review, msg_history
    else:
        return review


reviewer_reflection_prompt = load_prompt(
    "perform_llm_review.reviewer_reflection_prompt"
)


def load_paper(pdf_path, num_pages=None, min_size=100):
    try:
        if num_pages is None:
            text = pymupdf4llm.to_markdown(pdf_path)
        else:
            reader = PdfReader(pdf_path)
            min_pages = min(len(reader.pages), num_pages)
            text = pymupdf4llm.to_markdown(pdf_path, pages=list(range(min_pages)))
        if len(text) < min_size:
            raise Exception("Text too short")
    except Exception as e:
        print(f"Error with pymupdf4llm, falling back to pymupdf: {e}")
        try:
            doc = pymupdf.open(pdf_path)
            if num_pages:
                doc = doc[:num_pages]
            text = ""
            for page in doc:
                text += page.get_text()
            if len(text) < min_size:
                raise Exception("Text too short")
        except Exception as e:
            print(f"Error with pymupdf, falling back to pypdf: {e}")
            reader = PdfReader(pdf_path)
            if num_pages is None:
                pages = reader.pages
            else:
                pages = reader.pages[:num_pages]
            text = "".join(page.extract_text() for page in pages)
            if len(text) < min_size:
                raise Exception("Text too short")
    return text


def load_review(json_path):
    with open(json_path, "r") as json_file:
        loaded = json.load(json_file)
    return loaded["review"]


dir_path = os.path.dirname(os.path.realpath(__file__))

fewshot_papers = [
    os.path.join(dir_path, "fewshot_examples/132_automated_relational.pdf"),
    os.path.join(dir_path, "fewshot_examples/attention.pdf"),
    os.path.join(dir_path, "fewshot_examples/2_carpe_diem.pdf"),
]

fewshot_reviews = [
    os.path.join(dir_path, "fewshot_examples/132_automated_relational.json"),
    os.path.join(dir_path, "fewshot_examples/attention.json"),
    os.path.join(dir_path, "fewshot_examples/2_carpe_diem.json"),
]


def get_review_fewshot_examples(num_fs_examples=1):
    fewshot_prompt = load_prompt("perform_llm_review.fewshot_prompt_intro")
    for paper_path, review_path in zip(
        fewshot_papers[:num_fs_examples], fewshot_reviews[:num_fs_examples]
    ):
        txt_path = paper_path.replace(".pdf", ".txt")
        if os.path.exists(txt_path):
            with open(txt_path, "r") as f:
                paper_text = f.read()
        else:
            paper_text = load_paper(paper_path)
        review_text = load_review(review_path)
        fewshot_prompt += render_prompt(
            "perform_llm_review.fewshot_prompt_entry",
            paper_text=paper_text,
            review_text=review_text,
        )
    return fewshot_prompt


meta_reviewer_system_prompt = load_prompt(
    "perform_llm_review.meta_reviewer_system_prompt"
)


def get_meta_review(model, client, temperature, reviews):
    review_text = ""
    for i, r in enumerate(reviews):
        review_text += render_prompt(
            "perform_llm_review.meta_review_entry",
            review_index=i + 1,
            review_count=len(reviews),
            review_json=json.dumps(r),
        )
    base_prompt = neurips_form + review_text
    llm_review, _ = get_response_from_llm(
        base_prompt,
        model=model,
        client=client,
        system_message=render_prompt(
            "perform_llm_review.meta_reviewer_system_prompt",
            reviewer_count=len(reviews),
        ),
        print_debug=False,
        msg_history=None,
        temperature=temperature,
    )
    meta_review = extract_json_between_markers(llm_review)
    return meta_review
