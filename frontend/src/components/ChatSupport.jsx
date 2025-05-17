import React, { useState, useRef, useEffect } from "react";
import axios from "axios";
import { Textarea } from "./ui/textarea";
import { Button } from "./ui/button";
import { Send } from "lucide-react";

const BASE_API_URL = "http://127.0.0.1:8000"; // Update as needed

export default function ChatSupport() {
  const [messages, setMessages] = useState([
    { role: "bot", text: "Hello! How can I assist you today?" },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userText = input.trim();
    setMessages((prev) => [...prev, { role: "user", text: userText }]);
    setInput("");
    setLoading(true);

    try {
      const res = await axios.post(`${BASE_API_URL}/chat_with_report/`, { message: userText });

      const botReply = res?.data?.response || "Sorry, I didn't get that.";
      setMessages((prev) => [...prev, { role: "bot", text: botReply }]);
    } catch (err) {
      console.error(err);
      setMessages((prev) => [
        ...prev,
        { role: "bot", text: "Something went wrong. Please try again." },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed bottom-20 right-4 z-50 w-[95%] sm:w-96 h-[400px] rounded-lg border bg-background shadow-xl flex flex-col">
      {/* Header */}
      <div className="p-4 border-b">
        <h4 className="text-lg font-semibold">Live Chat</h4>
      </div>

      {/* Messages (Scrollable) */}
      <div className="flex-1 overflow-y-auto px-3 py-2 space-y-2">
        {messages.map((msg, i) => (
          <div
            key={i}
            className={`w-fit max-w-[80%] rounded-lg px-4 py-2 text-sm break-words ${
              msg.role === "user"
                ? "ml-auto bg-primary text-primary-foreground"
                : "mr-auto bg-muted text-muted-foreground"
            }`}
          >
            {msg.text}
          </div>
        ))}
        {loading && (
          <div className="mr-auto rounded-lg bg-muted px-4 py-2 text-sm text-muted-foreground">
            Typing...
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <form onSubmit={handleSubmit} className="flex items-end gap-2 border-t p-3">
        <Textarea
          placeholder="Type your message..."
          className="flex-grow resize-none h-10"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          rows={1}
          disabled={loading}
        />
        <Button type="submit" size="icon" disabled={loading || !input.trim()}>
          <Send className="h-4 w-4" />
          <span className="sr-only">Send</span>
        </Button>
      </form>
    </div>
  );
}
