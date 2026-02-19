import { useChatStore } from '../store/chatStore';
import { useModels, useStrategies } from '../hooks/useChat';

export default function ModelSelector() {
  const { settings, updateSettings } = useChatStore();
  const { data: modelsData } = useModels();
  const { data: strategiesData } = useStrategies();

  return (
    <div className="space-y-3 p-4">
      <div>
        <label className="mb-1 block text-xs font-medium text-zinc-400">Model</label>
        <select
          value={settings.model_name}
          onChange={(e) => updateSettings({ model_name: e.target.value })}
          className="w-full rounded-lg border border-zinc-600 bg-zinc-800 px-3 py-2 text-sm text-zinc-200 outline-none focus:border-blue-500"
        >
          {modelsData?.models?.map((m: { name: string }) => (
            <option key={m.name} value={m.name}>
              {m.name}
            </option>
          ))}
        </select>
      </div>

      <div>
        <label className="mb-1 block text-xs font-medium text-zinc-400">Strategy</label>
        <select
          value={settings.synthesis_strategy}
          onChange={(e) => updateSettings({ synthesis_strategy: e.target.value })}
          className="w-full rounded-lg border border-zinc-600 bg-zinc-800 px-3 py-2 text-sm text-zinc-200 outline-none focus:border-blue-500"
        >
          {strategiesData?.strategies?.map((s: string) => (
            <option key={s} value={s}>
              {s}
            </option>
          ))}
        </select>
      </div>

      <div className="flex items-center justify-between">
        <label className="text-xs font-medium text-zinc-400">Use RAG</label>
        <button
          onClick={() => updateSettings({ use_rag: !settings.use_rag })}
          className={`relative h-6 w-11 rounded-full transition-colors ${
            settings.use_rag ? 'bg-blue-600' : 'bg-zinc-600'
          }`}
        >
          <span
            className={`absolute left-0.5 top-0.5 h-5 w-5 rounded-full bg-white transition-transform ${
              settings.use_rag ? 'translate-x-5' : ''
            }`}
          />
        </button>
      </div>
    </div>
  );
}
