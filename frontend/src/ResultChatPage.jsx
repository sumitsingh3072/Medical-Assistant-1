import React, { useState } from 'react';
import axios from 'axios';
import { Input } from './components/ui/input';
import { Button } from './components/ui/button';
import { Card, CardContent } from './components/ui/card';

const BASE_API_URL = 'http://127.0.0.1:8000';

const ResultChatPage = () => {
  const [messages, setMessages] = useState([
    {
      role: 'ai',
      text: "Hi! I'm your report assistant. How can I help you with your diagnosis?",
    },
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSend = async () => {
    if (!input.trim()) return;

    const userMessage = { role: 'user', text: input };
    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    try {
      const res = await axios.post(`${BASE_API_URL}/chat_with_report/`, { message: input });

      const aiReply = res.data.response || 'Sorry, I could not understand that.';
      setMessages((prev) => [...prev, { role: 'ai', text: aiReply }]);
    } catch (err) {
      console.error(err);
      setMessages((prev) => [
        ...prev,
        { role: 'ai', text: 'There was an error fetching a response. Please try again.' },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-background text-foreground p-4 md:p-6 flex flex-col items-center">
      <Card className="w-full max-w-3xl flex flex-col flex-grow border border-border shadow-lg">
        <CardContent className="p-4 space-y-4 flex flex-col flex-grow">
          <h2 className="text-2xl font-semibold text-center">Report Chat Assistant</h2>

          <div className="flex-grow overflow-y-auto max-h-[60vh] space-y-3 p-2 border rounded-md bg-muted/30">
            {messages.map((msg, idx) => (
              <div
                key={idx}
                className={`p-3 rounded-md text-sm max-w-[80%] ${
                  msg.role === 'ai'
                    ? 'bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-100 font-semibold self-start'
                    : 'bg-zinc-800 dark:bg-white text-white dark:text-black font-semibold self-end ml-auto'
                }`}
              >
                {msg.text}
              </div>
            ))}
          </div>

          <form
            className="flex gap-2 pt-4"
            onSubmit={(e) => {
              e.preventDefault();
              handleSend();
            }}
          >
            <Input
              className="flex-grow"
              placeholder="Ask about your diagnosis..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              disabled={loading}
            />
            <Button type="submit" disabled={loading}>
              {loading ? 'Thinking...' : 'Send'}
            </Button>
          </form>
        </CardContent>
      </Card>
    </div>
  );
};

export default ResultChatPage;
