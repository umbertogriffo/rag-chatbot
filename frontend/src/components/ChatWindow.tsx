import React from 'react';
import { Paper, Box, Typography, Avatar } from '@mui/material';
import SmartToyIcon from '@mui/icons-material/SmartToy';
import PersonIcon from '@mui/icons-material/Person';
import Markdown from "react-markdown";


interface Message {
  text: string;
  sender: 'user' | 'bot';
  timestamp: Date;
}

interface ChatWindowProps {
  messages: Message[];
}

const MessageBubble: React.FC<Message> = ({ text, sender, timestamp }: Message) => (
  <Box
    sx={{
      display: 'flex',
      alignItems: 'flex-start',
      gap: 1,
      mb: 2,
      flexDirection: 'row',
    }}
  >
    <Avatar sx={{ bgcolor: sender === 'user' ? 'primary.main' : 'secondary.main' }}>
      {sender === 'user' ? <PersonIcon /> : <SmartToyIcon />}
    </Avatar>
    <Box
      sx={{
        maxWidth: '100%',
        p: 2,
        borderRadius: 2,
        bgcolor: sender === 'user' ? 'primary.light' : 'grey.100', textAlign: 'left',
      }}
    >
        <Markdown>{text}</Markdown>
      <Typography variant="caption" sx={{ display: 'block', mt: 1, opacity: 0.7 }}>
        {timestamp.toLocaleTimeString()}
      </Typography>
    </Box>
  </Box>
);


const ChatWindow: React.FC<ChatWindowProps> = ({ messages }) => {

const messagesEndRef = React.useRef<null | HTMLDivElement>(null);

  React.useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  if (messages.length === 0) {
    return null;
  }

  return (
    <Paper
      sx={{
        mt: 2,
        p: 2,
        height: '60vh',
        overflow: 'auto',
        width: '100%',
        '&::-webkit-scrollbar': {
          width: '8px',
        },
        '&::-webkit-scrollbar-track': {
          background: '#f1f1f1',
          borderRadius: '4px',
        },
        '&::-webkit-scrollbar-thumb': {
          background: '#888',
          borderRadius: '4px',
          '&:hover': {
            background: '#555',
          },
        },
      }}>
      {messages.map((message, index) => (
        <MessageBubble key={index} {...message} />
      ))}
      <div ref={messagesEndRef} />
    </Paper>
  );
};

export type { Message };
export default ChatWindow;
