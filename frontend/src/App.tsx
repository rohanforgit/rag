import { useState, useEffect, useRef } from 'react'
import { 
  MessageSquare, 
  Send, 
  Plus, 
  Trash2, 
  BookOpen, 
  FileText 
} from 'lucide-react'

interface Source {
  page: number;
  section: string;
  document: string;
}

const formatDocName = (doc: string): string => {
  if (!doc) return "Regulations";
  const name = doc.replace(/\.pdf$/i, '').replace(/\.xlsx$/i, '').replace(/\.xls$/i, '');
  if (name.toLowerCase().includes("rules_regulations") || name.toLowerCase().includes("rules & regulations")) {
    return "R26 Regulations";
  }
  if (name.toLowerCase().includes("cse_curriculum")) {
    return "CSE Syllabus";
  }
  if (name.toLowerCase().includes("aiml_curriculum")) {
    return "AIML Syllabus";
  }
  if (name.toLowerCase().includes("calendar") && name.toLowerCase().includes("odd")) {
    return "Calendar (Odd)";
  }
  if (name.toLowerCase().includes("calendar") && name.toLowerCase().includes("even")) {
    return "Calendar (Even)";
  }
  if (name.toLowerCase().includes("evaluation_structures")) {
    return "Evaluation Structure";
  }
  return name.replace(/_/g, ' ');
};

interface Message {
  role: 'user' | 'assistant';
  content: string;
  sources?: Source[];
}

interface Session {
  id: string;
  name: string;
}

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 
  (typeof window !== 'undefined' && window.location.hostname === 'localhost' ? 'http://localhost:8000' : '/api');

function App() {
  const [sessions, setSessions] = useState<Session[]>(() => {
    const saved = localStorage.getItem('sreenidhi_sessions');
    return saved ? JSON.parse(saved) : [];
  });
  
  const [activeSessionId, setActiveSessionId] = useState<string>(() => {
    const savedActive = localStorage.getItem('sreenidhi_active_session');
    return savedActive || '';
  });

  const [messagesBySession, setMessagesBySession] = useState<Record<string, Message[]>>(() => {
    const savedMessages = localStorage.getItem('sreenidhi_messages');
    return savedMessages ? JSON.parse(savedMessages) : {};
  });

  const [input, setInput] = useState('');
  const [isThinking, setIsThinking] = useState(false);
  const [apiOnline, setApiOnline] = useState(false);
  
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Check backend health status
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/health`);
        if (res.ok) {
          setApiOnline(true);
        } else {
          setApiOnline(false);
        }
      } catch {
        setApiOnline(false);
      }
    };

    checkHealth();
    const interval = setInterval(checkHealth, 10000);
    return () => clearInterval(interval);
  }, []);

  // Save sessions to localStorage
  useEffect(() => {
    localStorage.setItem('sreenidhi_sessions', JSON.stringify(sessions));
    localStorage.setItem('sreenidhi_active_session', activeSessionId);
    localStorage.setItem('sreenidhi_messages', JSON.stringify(messagesBySession));
  }, [sessions, activeSessionId, messagesBySession]);

  // Scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messagesBySession, activeSessionId, isThinking]);

  // Initialize a new session if none exists
  useEffect(() => {
    if (!activeSessionId && sessions.length === 0) {
      handleNewChat();
    }
  }, []);

  const handleNewChat = () => {
    const newId = crypto.randomUUID();
    const newSession: Session = {
      id: newId,
      name: `Chat Session ${sessions.length + 1}`
    };
    
    setSessions(prev => [newSession, ...prev]);
    setActiveSessionId(newId);
    setMessagesBySession(prev => ({
      ...prev,
      [newId]: []
    }));
  };

  const handleDeleteSession = async (e: React.MouseEvent, sessionId: string) => {
    e.stopPropagation();
    
    // Call backend to clear memory
    try {
      await fetch(`${API_BASE_URL}/chat/${sessionId}`, { method: 'DELETE' });
    } catch (err) {
      console.warn("Could not notify backend of deleted session:", err);
    }

    const updatedSessions = sessions.filter(s => s.id !== sessionId);
    setSessions(updatedSessions);
    
    const updatedMessages = { ...messagesBySession };
    delete updatedMessages[sessionId];
    setMessagesBySession(updatedMessages);

    if (activeSessionId === sessionId) {
      if (updatedSessions.length > 0) {
        setActiveSessionId(updatedSessions[0].id);
      } else {
        setActiveSessionId('');
      }
    }
  };

  const handleSendMessage = async (textToSend: string) => {
    const trimmed = textToSend.trim();
    if (!trimmed || isThinking) return;

    let currentSessionId = activeSessionId;
    
    // Create new session if none exists
    if (!currentSessionId) {
      const newId = crypto.randomUUID();
      const newSession: Session = {
        id: newId,
        name: trimmed.length > 25 ? trimmed.substring(0, 25) + '...' : trimmed
      };
      setSessions([newSession]);
      setActiveSessionId(newId);
      setMessagesBySession({ [newId]: [] });
      currentSessionId = newId;
    }

    const userMessage: Message = {
      role: 'user',
      content: trimmed
    };

    // Update messages locally with User message
    setMessagesBySession(prev => ({
      ...prev,
      [currentSessionId]: [...(prev[currentSessionId] || []), userMessage]
    }));

    setInput('');
    setIsThinking(true);

    try {
      const response = await fetch(`${API_BASE_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: trimmed,
          session_id: currentSessionId
        })
      });

      if (!response.ok) {
        throw new Error(await response.text() || 'Failed to communicate with API');
      }

      const data = await response.json();
      
      const assistantMessage: Message = {
        role: 'assistant',
        content: data.answer,
        sources: data.sources
      };

      // Rename session based on first message if default name
      setSessions(prev => 
        prev.map(s => {
          if (s.id === currentSessionId && s.name.startsWith('Chat Session')) {
            return {
              ...s,
              name: trimmed.length > 22 ? trimmed.substring(0, 22) + '...' : trimmed
            };
          }
          return s;
        })
      );

      setMessagesBySession(prev => ({
        ...prev,
        [currentSessionId]: [...(prev[currentSessionId] || []), assistantMessage]
      }));

    } catch (error: any) {
      console.error(error);
      const errorMessage: Message = {
        role: 'assistant',
        content: `Error: Unable to fetch regulations. Please check if backend FastAPI server is running. (Details: ${error.message})`
      };
      setMessagesBySession(prev => ({
        ...prev,
        [currentSessionId]: [...(prev[currentSessionId] || []), errorMessage]
      }));
    } finally {
      setIsThinking(false);
    }
  };

  const handleFormSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    handleSendMessage(input);
  };

  const currentMessages = activeSessionId ? (messagesBySession[activeSessionId] || []) : [];

  const suggestions = [
    {
      title: "Attendance Policy",
      desc: "What is the minimum attendance required and condonation guidelines?",
      query: "What is the attendance policy and how do medical leaves work?"
    },
    {
      title: "Supplementary Exams",
      desc: "How do supply exams, arrears, and revaluation process function?",
      query: "What are the rules for supplementary exams and revaluation?"
    },
    {
      title: "Minors and Honors",
      desc: "What is the criteria to register and earn a Minor or Honors degree?",
      query: "What are the eligibility criteria and rules for Minor and Honors programs?"
    },
    {
      title: "Promotion Regulations",
      desc: "What are the requirements for credit promotion to subsequent years?",
      query: "What are the promotion rules and credit requirements for B.Tech?"
    }
  ];

  return (
    <div className="app-container">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="sidebar-header">
          <div className="logo-icon">S</div>
          <div className="logo-text">Sreenidhi AI</div>
        </div>

        <button className="new-chat-btn" onClick={handleNewChat}>
          <Plus size={18} />
          New Discussion
        </button>

        <div className="sessions-list">
          {sessions.map(s => (
            <div
              key={s.id}
              className={`session-item ${activeSessionId === s.id ? 'active' : ''}`}
              onClick={() => setActiveSessionId(s.id)}
              role="button"
              tabIndex={0}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  setActiveSessionId(s.id);
                }
              }}
            >
              <div className="session-info">
                <MessageSquare size={16} />
                <span>{s.name}</span>
              </div>
              <button 
                className="delete-session-btn" 
                onClick={(e) => handleDeleteSession(e, s.id)}
                title="Delete chat session"
              >
                <Trash2 size={14} />
              </button>
            </div>
          ))}
        </div>

        <div className="sidebar-footer">
          R26 Regulations v1.0
        </div>
      </aside>

      {/* Main Chat Panel */}
      <main className="chat-main">
        {/* Header */}
        <header className="chat-header">
          <div className="header-title-container">
            <h1 className="header-title">Academic Regulations Assistant</h1>
            <span className="header-subtitle">Official Sreenidhi B.Tech R26 Guidelines</span>
          </div>

          <div className="api-status-badge" style={{ 
            backgroundColor: apiOnline ? 'rgba(16, 185, 129, 0.1)' : 'rgba(239, 68, 68, 0.1)',
            color: apiOnline ? '#10b981' : '#f87171',
            borderColor: apiOnline ? 'rgba(16, 185, 129, 0.2)' : 'rgba(239, 68, 68, 0.2)'
          }}>
            <span className="status-dot" style={{ 
              backgroundColor: apiOnline ? '#10b981' : '#f87171',
              boxShadow: apiOnline ? '0 0 8px #10b981' : '0 0 8px #f87171'
            }}></span>
            {apiOnline ? 'Server Online' : 'Server Offline'}
          </div>
        </header>

        {/* Message Feed / Welcome Screen */}
        {currentMessages.length === 0 ? (
          <div className="welcome-container">
            <div className="welcome-icon">
              <BookOpen size={48} color="#8b5cf6" />
            </div>
            <h2 className="welcome-title">How can I help you today?</h2>
            <p className="welcome-desc">
              Ask any question about the official Sreenidhi University B.Tech R26 rules and regulations. 
              The response will be strictly grounded in the official handbook with exact sources provided.
            </p>

            <div className="suggestions-grid">
              {suggestions.map((s, idx) => (
                <div 
                  key={idx} 
                  className="suggestion-card"
                  onClick={() => handleSendMessage(s.query)}
                >
                  <h4>{s.title}</h4>
                  <p>{s.desc}</p>
                </div>
              ))}
            </div>
          </div>
        ) : (
          <div className="messages-feed">
            {currentMessages.map((m, idx) => (
              <div key={idx} className={`message-wrapper ${m.role}`}>
                <div className={`avatar ${m.role}`}>
                  {m.role === 'user' ? 'U' : 'AI'}
                </div>
                <div className="message-bubble">
                  <p style={{ whiteSpace: 'pre-wrap' }}>{m.content}</p>
                  
                  {/* Sources tag list */}
                  {m.role === 'assistant' && m.sources && m.sources.length > 0 && (
                    <div className="sources-container">
                      <span className="sources-label">Sources:</span>
                      <div className="sources-list">
                        {m.sources.map((src, sIdx) => (
                          <div key={sIdx} className="source-badge" title={`${src.document || 'Regulations'} - ${src.section} (Page ${src.page})`}>
                            <FileText size={12} />
                            <span>
                              [{formatDocName(src.document)}] {
                                (src.document && (src.document.toLowerCase().endsWith('.xlsx') || src.document.toLowerCase().endsWith('.xls')))
                                  ? src.section
                                  : `Page ${src.page} — ${src.section}`
                              }
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            ))}
            
            {/* Thinking Indicator */}
            {isThinking && (
              <div className="message-wrapper assistant">
                <div className="avatar assistant">AI</div>
                <div className="message-bubble">
                  <div className="typing-dots">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                </div>
              </div>
            )}
            
            <div ref={messagesEndRef} />
          </div>
        )}

        {/* Input Bar */}
        <footer className="input-container">
          <form className="input-form" onSubmit={handleFormSubmit}>
            <input
              type="text"
              className="chat-input"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask SreenidhiAI about regulations..."
              disabled={isThinking}
            />
            <button 
              type="submit" 
              className="send-btn"
              disabled={!input.trim() || isThinking}
            >
              <Send size={18} />
            </button>
          </form>
          <p className="input-disclaimer">
            Answers are grounded solely in official Sreenidhi R26 academic documents.
          </p>
        </footer>
      </main>
    </div>
  )
}

export default App
