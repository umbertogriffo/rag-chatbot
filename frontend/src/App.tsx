import './App.css'
import {Container, CssBaseline, ThemeProvider, createTheme, Box} from '@mui/material';
import ChatWindow from "./components/ChatWindow.tsx";
import SearchBar from "./components/SearchBar.tsx";
import React from "react";
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'

const theme = createTheme({
    palette: {
        mode: 'light',
        primary: {
            main: '#10a37f', // green
            contrastText: '#fff'
        },
        background: {
            default: '#fff',
            paper: '#fff'
        },
        text: {
            primary: '#000',
            secondary: '#6e6e80'
        }
    },
    components: {
        MuiButton: {
            styleOverrides: {
                root: {
                    textTransform: 'none',
                    borderRadius: 8,
                    padding: '8px 16px'
                }
            }
        },
        MuiPaper: {
            styleOverrides: {
                root: {
                    borderRadius: 12,
                    boxShadow: '0 2px 6px rgba(0, 0, 0, 0.05)'
                }
            }
        },
        MuiTextField: {
            styleOverrides: {
                root: {
                    '& .MuiOutlinedInput-root': {
                        borderRadius: 8
                    }
                }
            }
        }
    },
    shape: {
        borderRadius: 8
    }
});


function App() {
    console.log('log');
    const [messages, setMessages] = React.useState<Array<{
        text: string;
        sender: 'user' | 'bot';
        timestamp: Date;
    }>>([]);

    const handleSearch = async (query: string, options: {
        reasoning: boolean,
        webSearch: boolean
    }) => {
        console.log('Search query:', query, 'Options:', options);
        // Add user message
        setMessages(prev => [...prev, {
            text: query,
            sender: 'user',
            timestamp: new Date()
        }]);

        try {
            const response = await fetch(`${import.meta.env.VITE_API_URL}/api/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    text: query,
                    reasoning: options.reasoning,
                    web_search: options.webSearch,
                }),
            });
            const data = await response.json();
            console.log('answer:', data);
            // Add bot response
            setMessages(prev => [...prev, {
                text: data.response,
                sender: 'bot',
                timestamp: new Date()
            }]);
        } catch (error) {
            console.error('Error:', error);
        }
    };

    return (
        <>
            <div>
                <a href="https://vite.dev" target="_blank">
                    <img src={viteLogo} className="logo" alt="Vite logo"/>
                </a>
                <a href="https://react.dev" target="_blank">
                    <img src={reactLogo} className="logo react" alt="React logo"/>
                </a>
            </div>
            <h2>What can I help with?</h2>
            <ThemeProvider theme={theme}>
            <CssBaseline/>
            <Container
                maxWidth={false}
                sx={{
                    mt: 4,
                    maxWidth: '100vw',
                    width: '55vw',
                    marginLeft: 0,
                    marginRight: 0,
                    paddingLeft: 0,
                    paddingRight: 0
                }}>
                <Box sx={{
                    width: '100%',
                    mx: 'auto',
                    px: {xs: 2, lg: 4}
                }}>
                    <ChatWindow messages={messages}/>
                    <SearchBar onSearch={handleSearch}/>
                </Box>
            </Container>
        </ThemeProvider></>
    );
}

export default App
