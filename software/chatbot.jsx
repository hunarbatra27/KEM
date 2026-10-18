
import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './chatbot.css';

// Local Flask backend
const API_URL = 'http://192.168.1.64:5000';

const Chatbot = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);

  const navigate = useNavigate();

  const sendMessage = async () => {
    const question = input.trim();

    if (!question || loading) return;

    const userMessage = {
      sender: 'user',
      text: question,
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    try {
      const response = await fetch(`${API_URL}/ask`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: question,
        }),
      });

      // Read the response even when the server returns an error
      const data = await response.json().catch(() => ({}));

      if (!response.ok) {
        throw new Error(
          `Server returned ${response.status}: ${
            data.error || response.statusText
          }`
        );
      }

      const botMessage = {
        sender: 'bot',
        text: data.answer || 'No answer received.',
      };

      setMessages((prev) => [...prev, botMessage]);
    } catch (error) {
      console.error('Chatbot API error:', error);

      setMessages((prev) => [
        ...prev,
        {
          sender: 'bot',
          text: `⚠️ ${error.message || 'Error contacting server.'}`,
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      sendMessage();
    }
  };

  return (
    <div className="chatbot-container">
      <h1 className="chatbot-title">Ask Me Anything</h1>

      <div className="chat-window">
        {messages.map((msg, index) => (
          <div
            key={index}
            className={`message ${msg.sender}`}
          >
            {msg.text}
          </div>
        ))}

        {loading && (
          <div className="message bot">
            Thinking...
          </div>
        )}
      </div>

      <div className="input-bar">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyPress}
          placeholder="Type your question..."
          disabled={loading}
        />

        <button onClick={sendMessage} disabled={loading}>
          {loading ? 'Sending...' : 'Send'}
        </button>
      </div>

      <div className="back-btn-wrapper">
        <button
          className="back-btn"
          onClick={() => navigate('/')}
        >
          ⬅ Back to Home
        </button>
      </div>
    </div>
  );
};

export default Chatbot;
