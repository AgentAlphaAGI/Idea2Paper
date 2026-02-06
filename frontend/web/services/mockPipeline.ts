/**
 * DEPRECATED
 * 
 * The mock logic has been consolidated into services/api.ts to provide a unified 
 * interface for both Real and Mock modes via configuration.
 * 
 * Please refer to 'services/api.ts' for the implementation.
 */

export const runMockPipeline = async () => {
    throw new Error("This module is deprecated. Use api.runPipeline from services/api.ts instead.");
};
