import React, { useState } from 'react';
import {
    TextField,
    IconButton,
    Paper,
    Box
} from '@mui/material';
import SearchIcon from '@mui/icons-material/Search';
import PsychologyIcon from '@mui/icons-material/Psychology';
import LanguageIcon from '@mui/icons-material/Language';

interface SearchBarProps {
    onSearch: (query: string, options: SearchOptions) => void;
}

interface SearchOptions {
    reasoning: boolean;
    webSearch: boolean;
}

export const SearchBar: React.FC<SearchBarProps> = ({ onSearch }) => {
    const [query, setQuery] = useState('');
    const [options, setOptions] = useState<SearchOptions>({
        reasoning: false,
        webSearch: false,
    });

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        if (query.trim()) {
            onSearch(query, options);
            setQuery(''); // Clear input after submission
        }
    };

    const toggleOption = (option: keyof SearchOptions) => {
        setOptions(prev => ({
            ...prev,
            [option]: !prev[option]
        }));
    };

return (
        <Paper
            component="form"
            onSubmit={handleSubmit}
            sx={{
                p: '2px 4px',
                display: 'flex',
                alignItems: 'center',
                width: '100%',
            }}
        >
            <TextField
                sx={{ ml: 1, flex: 1, mx: 'auto'}}
                placeholder="Ask anything"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                variant="standard"
                multiline
                maxRows={4}
                InputProps={{
                    disableUnderline: true,
                    sx: { alignItems: 'flex-start' }
            }}
            />
            <Box sx={{ display: 'flex', gap: 1, mx: 'auto' }}>
                <IconButton
                    color={options.reasoning ? "primary" : "default"}
                    onClick={() => toggleOption('reasoning')}
                    size="small"
                    type="button"
                >
                    <PsychologyIcon />
                </IconButton>
                <IconButton
                    color={options.webSearch ? "primary" : "default"}
                    onClick={() => toggleOption('webSearch')}
                    size="small"
                    type="button"
                >
                    <LanguageIcon />
                </IconButton>
                <IconButton type="submit" size="small">
                    <SearchIcon />
                </IconButton>
            </Box>
        </Paper>
    );
};

export default SearchBar;
