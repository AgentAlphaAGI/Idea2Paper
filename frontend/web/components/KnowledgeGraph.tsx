import React, { useState, useEffect, useCallback } from 'react';
import ReactFlow, {
  Node,
  Edge,
  Controls,
  Background,
  useNodesState,
  useEdgesState,
  MarkerType,
  Panel,
} from 'reactflow';
import 'reactflow/dist/style.css';
import { AppConfig } from '../types';
import { Loader2, AlertCircle, Database } from 'lucide-react';

interface KnowledgeGraphProps {
  config: AppConfig;
  t: any;
}

interface KGNode {
  paper_id: string;
  title: string;
  domain: string;
  sub_domains: string[];
  cluster_id: number;
  review_stats?: {
    avg_score: number;
    review_count: number;
  };
}

interface KGEdge {
  source: string;
  target: string;
  relation: string;
  quality?: number;
}

interface KGData {
  nodes: KGNode[];
  edges: KGEdge[];
  total_nodes: number;
  total_edges: number;
}

export const KnowledgeGraph: React.FC<KnowledgeGraphProps> = ({ config, t }) => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [kgData, setKgData] = useState<KGData | null>(null);
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);

  // Fetch KG data from backend
  useEffect(() => {
    const fetchKGData = async () => {
      try {
        setLoading(true);
        setError(null);

        const response = await fetch(`${config.baseUrl}/api/kg`);
        const data = await response.json();

        if (!data.ok) {
          throw new Error(data.error || 'Failed to load knowledge graph data');
        }

        setKgData(data);

        // Transform nodes for ReactFlow
        const flowNodes: Node[] = data.nodes.map((node: KGNode, index: number) => {
          // Calculate position in a circular layout
          const angle = (index / data.nodes.length) * 2 * Math.PI;
          const radius = 400;
          const x = Math.cos(angle) * radius + 500;
          const y = Math.sin(angle) * radius + 400;

          // Color by domain
          const domainColors: Record<string, string> = {
            'Computer Vision': '#8b5cf6',
            'Natural Language Processing': '#3b82f6',
            'Reinforcement Learning': '#10b981',
            'Fairness & Accountability': '#f59e0b',
            'default': '#6b7280'
          };

          const color = domainColors[node.domain] || domainColors.default;

          return {
            id: node.paper_id,
            type: 'default',
            position: { x, y },
            data: {
              label: node.title.length > 50 ? node.title.substring(0, 50) + '...' : node.title,
            },
            style: {
              background: color,
              color: 'white',
              border: '2px solid white',
              borderRadius: '8px',
              padding: '10px',
              fontSize: '12px',
              width: 200,
            },
          };
        });

        // Transform edges for ReactFlow
        const flowEdges: Edge[] = data.edges.map((edge: KGEdge, index: number) => ({
          id: `edge-${index}`,
          source: edge.source,
          target: edge.target,
          label: edge.relation,
          type: 'smoothstep',
          animated: edge.relation === 'implements',
          style: { stroke: '#94a3b8', strokeWidth: 1 },
          markerEnd: {
            type: MarkerType.ArrowClosed,
            color: '#94a3b8',
          },
        }));

        setNodes(flowNodes);
        setEdges(flowEdges);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unknown error');
      } finally {
        setLoading(false);
      }
    };

    fetchKGData();
  }, [config.baseUrl, setNodes, setEdges]);

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center h-[70vh] bg-white dark:bg-slate-800 rounded-3xl border border-slate-200 dark:border-slate-700">
        <Loader2 className="w-12 h-12 text-violet-500 animate-spin mb-4" />
        <p className="text-slate-600 dark:text-slate-300 font-medium">Loading Knowledge Graph...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center h-[70vh] bg-white dark:bg-slate-800 rounded-3xl border border-slate-200 dark:border-slate-700 border-dashed">
        <AlertCircle className="w-12 h-12 text-red-500 mb-4" />
        <h3 className="text-xl font-bold text-slate-800 dark:text-slate-200 mb-2">Failed to Load Knowledge Graph</h3>
        <p className="text-slate-500 dark:text-slate-400 max-w-md text-center">{error}</p>
      </div>
    );
  }

  return (
    <div className="h-[80vh] bg-white dark:bg-slate-800 rounded-3xl border border-slate-200 dark:border-slate-700 overflow-hidden">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        fitView
        attributionPosition="bottom-left"
      >
        <Background color="#94a3b8" gap={16} />
        <Controls />
        <Panel position="top-left" className="bg-white dark:bg-slate-800 p-4 rounded-xl border border-slate-200 dark:border-slate-700 shadow-lg">
          <div className="flex items-center gap-3 mb-3">
            <Database className="w-5 h-5 text-violet-500" />
            <h3 className="font-bold text-slate-800 dark:text-white">Knowledge Graph</h3>
          </div>
          {kgData && (
            <div className="text-xs text-slate-600 dark:text-slate-300 space-y-1">
              <p>Showing {kgData.nodes.length} of {kgData.total_nodes} papers</p>
              <p>{kgData.edges.length} relationships</p>
            </div>
          )}
        </Panel>
        <Panel position="top-right" className="bg-white dark:bg-slate-800 p-4 rounded-xl border border-slate-200 dark:border-slate-700 shadow-lg">
          <h4 className="font-bold text-slate-800 dark:text-white text-sm mb-2">Legend</h4>
          <div className="space-y-2 text-xs">
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded" style={{ background: '#8b5cf6' }}></div>
              <span className="text-slate-600 dark:text-slate-300">Computer Vision</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded" style={{ background: '#3b82f6' }}></div>
              <span className="text-slate-600 dark:text-slate-300">NLP</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded" style={{ background: '#10b981' }}></div>
              <span className="text-slate-600 dark:text-slate-300">RL</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded" style={{ background: '#f59e0b' }}></div>
              <span className="text-slate-600 dark:text-slate-300">Fairness</span>
            </div>
          </div>
        </Panel>
      </ReactFlow>
    </div>
  );
};
