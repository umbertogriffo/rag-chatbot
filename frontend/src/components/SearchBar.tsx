import React, {useState} from 'react';
import {
    TextField,
    IconButton,
    Paper,
    Box
} from '@mui/material';
import SearchIcon from '@mui/icons-material/Search';
import PsychologyIcon from '@mui/icons-material/Psychology';
import LanguageIcon from '@mui/icons-material/Language';
import DocumentScannerIcon from "@mui/icons-material/DocumentScanner";

interface SearchBarProps {
    onSearch: (query: string, options: SearchOptions) => void;
}

interface SearchOptions {
    rag: boolean;
    reasoning: boolean;
    webSearch: boolean;
}

export const SearchBar: React.FC<SearchBarProps> = ({onSearch}) => {
    const [query, setQuery] = useState('');
    const [options, setOptions] = useState<SearchOptions>({
        rag: false,
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
                sx={{ml: 1, flex: 1, mx: 'auto'}}
                placeholder="Ask anything"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                variant="standard"
                multiline
                maxRows={4}
                InputProps={{
                    disableUnderline: true,
                    sx: {alignItems: 'flex-start'}
                }}
            />
            <Box sx={{display: 'flex', gap: 1, mx: 'auto'}}>
                <IconButton
                    color={options.rag ? "primary" : "default"}
                    onClick={() => toggleOption('rag')}
                    size="small"
                    type="button"
                >
                    <DocumentScannerIcon/>
                </IconButton>
                <IconButton
                    color={options.webSearch ? "primary" : "default"}
                    onClick={() => toggleOption('webSearch')}
                    size="small"
                    type="button"
                >
                    <LanguageIcon/>
                </IconButton>
                <IconButton
                    color={options.reasoning ? "primary" : "default"}
                    onClick={() => toggleOption('reasoning')}
                    size="small"
                    type="button"
                >
                    <PsychologyIcon/>
                </IconButton>
                <IconButton type="submit" size="small">
                    <SearchIcon/>
                </IconButton>
            </Box>
        </Paper>
    );
};

export default SearchBar;
