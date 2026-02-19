import { useChatStore } from '../store/chatStore';

export default function SettingsPanel() {
  const { settings, updateSettings } = useChatStore();

  return (
    <div className="space-y-3 border-t border-zinc-700 p-4">
      <h3 className="text-xs font-semibold uppercase tracking-wider text-zinc-500">Settings</h3>

      <div>
        <label className="mb-1 block text-xs text-zinc-400">Max Tokens</label>
        <input
          type="number"
          value={settings.max_new_tokens}
          onChange={(e) => updateSettings({ max_new_tokens: parseInt(e.target.value) || 512 })}
          min={64}
          max={4096}
          className="w-full rounded-lg border border-zinc-600 bg-zinc-800 px-3 py-2 text-sm text-zinc-200 outline-none focus:border-blue-500"
        />
      </div>

      <div>
        <label className="mb-1 block text-xs text-zinc-400">Retrieved Chunks (k)</label>
        <input
          type="number"
          value={settings.k}
          onChange={(e) => updateSettings({ k: parseInt(e.target.value) || 2 })}
          min={1}
          max={10}
          className="w-full rounded-lg border border-zinc-600 bg-zinc-800 px-3 py-2 text-sm text-zinc-200 outline-none focus:border-blue-500"
        />
      </div>
    </div>
  );
}
