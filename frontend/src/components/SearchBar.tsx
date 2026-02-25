import React, { useState } from 'react';

interface SearchBarProps {
  onSearch: (query: string, options: SearchOptions) => void;
  disabled?: boolean;
}

interface SearchOptions {
  rag: boolean;
  reasoning: boolean;
  webSearch: boolean;
}

const SearchBar: React.FC<SearchBarProps> = ({ onSearch, disabled }) => {
  const [query, setQuery] = useState('');
  const [options, setOptions] = useState<SearchOptions>({
    rag: false,
    reasoning: false,
    webSearch: false,
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim() && !disabled) {
      onSearch(query, options);
      setQuery('');
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e as unknown as React.FormEvent);
    }
  };

  const toggle = (key: keyof SearchOptions) =>
    setOptions((prev) => ({ ...prev, [key]: !prev[key] }));

  const btnBase =
    'rounded-full px-3 py-1 text-xs font-medium transition-colors select-none';
  const btnActive = `${btnBase} bg-green-600 text-white`;
  const btnInactive = `${btnBase} bg-gray-700 text-gray-400 hover:bg-gray-600`;

  return (
    <form
      onSubmit={handleSubmit}
      className="flex flex-col gap-2 rounded-2xl border border-gray-700 bg-gray-900 p-3 shadow-lg"
    >
      <textarea
        className="w-full resize-none bg-transparent text-gray-100 placeholder-gray-500 text-sm outline-none leading-relaxed"
        placeholder="Ask anything… (Shift+Enter for new line)"
        rows={2}
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        onKeyDown={handleKeyDown}
        disabled={disabled}
      />
      <div className="flex items-center justify-between gap-2">
        <div className="flex gap-2 flex-wrap">
          <button
            type="button"
            onClick={() => toggle('rag')}
            className={options.rag ? btnActive : btnInactive}
          >
            📄 RAG
          </button>
          <button
            type="button"
            onClick={() => toggle('webSearch')}
            className={options.webSearch ? btnActive : btnInactive}
          >
            🌐 Web
          </button>
          <button
            type="button"
            onClick={() => toggle('reasoning')}
            className={options.reasoning ? btnActive : btnInactive}
          >
            🧠 Reason
          </button>
        </div>
        <button
          type="submit"
          disabled={!query.trim() || disabled}
          className="flex-shrink-0 rounded-full bg-green-600 px-4 py-1.5 text-sm font-semibold text-white hover:bg-green-500 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        >
          {disabled ? '⟳' : '↑'}
        </button>
      </div>
    </form>
  );
};

export default SearchBar;
