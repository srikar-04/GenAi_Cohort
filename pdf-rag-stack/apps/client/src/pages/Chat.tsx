import { useEffect, useMemo, useState } from 'react';
import { useListDocumentsQuery } from '../features/documents/documentsApi';
import { useQueryRagMutation } from '../features/rag/ragApi';

type ChatMessage = {
  role: 'user' | 'assistant';
  content: string;
};

type ChatSession = {
  id: string;
  title: string;
  messages: ChatMessage[];
  createdAt: string;
};

export const Chat = () => {
  const { data } = useListDocumentsQuery();
  const documents = data?.documents ?? [];
  const [documentId, setDocumentId] = useState<string | undefined>(documents[0]?.id);
  const [question, setQuestion] = useState('');
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [queryRag, { isLoading }] = useQueryRagMutation();

  useEffect(() => {
    if (!documentId && documents.length > 0) {
      setDocumentId(documents[0].id);
    }
  }, [documents, documentId]);

  useEffect(() => {
    if (sessions.length === 0) {
      const id =
        globalThis.crypto?.randomUUID?.() ??
        `${Date.now()}-${Math.random().toString(16).slice(2)}`;
      const seed: ChatSession = {
        id,
        title: 'New chat',
        messages: [],
        createdAt: new Date().toISOString(),
      };
      setSessions([seed]);
      setActiveSessionId(id);
    }
  }, [sessions.length]);

  const activeSession = useMemo(
    () => sessions.find((session) => session.id === activeSessionId) ?? sessions[0],
    [sessions, activeSessionId],
  );

  const updateSessionMessages = (sessionId: string, updater: (items: ChatMessage[]) => ChatMessage[]) => {
    setSessions((prev) =>
      prev.map((session) => {
        if (session.id !== sessionId) return session;
        const nextMessages = updater(session.messages);
        const nextTitle =
          session.title === 'New chat'
            ? nextMessages.find((message) => message.role === 'user')?.content.slice(0, 40) ??
              session.title
            : session.title;
        return {
          ...session,
          title: nextTitle,
          messages: nextMessages,
        };
      }),
    );
  };

  const handleNewChat = () => {
    const id =
      globalThis.crypto?.randomUUID?.() ??
      `${Date.now()}-${Math.random().toString(16).slice(2)}`;
    const nextSession: ChatSession = {
      id,
      title: 'New chat',
      messages: [],
      createdAt: new Date().toISOString(),
    };
    setSessions((prev) => [nextSession, ...prev]);
    setActiveSessionId(id);
  };

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!question.trim()) return;
    if (!activeSession) return;
    const userMessage = { role: 'user' as const, content: question };
    updateSessionMessages(activeSession.id, (prev) => [...prev, userMessage]);
    setQuestion('');
    const response = await queryRag({ documentId, question }).unwrap();
    updateSessionMessages(activeSession.id, (prev) => [
      ...prev,
      { role: 'assistant', content: response.answer || 'No answer returned.' },
    ]);
  };

  return (
    <div className="page">
      <header className="page-header">
        <h2>Chat</h2>
        <p>Ask questions grounded in your indexed PDFs.</p>
      </header>
      <div className="chat-layout">
        <div className="chat-sidebar">
          <div className="chat-sidebar-section">
            <label htmlFor="document-select">Choose document</label>
            <select
              id="document-select"
              value={documentId ?? ''}
              onChange={(event) => setDocumentId(event.target.value)}
            >
              <option value="">All documents</option>
              {documents.map((doc) => (
                <option key={doc.id} value={doc.id}>
                  {doc.title}
                </option>
              ))}
            </select>
            <div className="chat-tip">
              Use the question box to target summaries, methods, or citations.
            </div>
          </div>
          <div className="chat-sidebar-section">
            <div className="chat-history-header">
              <h4>Message history</h4>
              <button className="btn ghost" type="button" onClick={handleNewChat}>
                New chat
              </button>
            </div>
            <div className="chat-history">
              {sessions.map((session) => (
                <button
                  key={session.id}
                  type="button"
                  className={`chat-history-item ${
                    session.id === activeSession?.id ? 'active' : ''
                  }`}
                  onClick={() => setActiveSessionId(session.id)}
                >
                  <span>{session.title}</span>
                  <small>{new Date(session.createdAt).toLocaleDateString()}</small>
                </button>
              ))}
            </div>
          </div>
        </div>
        <div className="chat-panel">
          <div className="chat-messages">
            {activeSession?.messages.length === 0 ? (
              <div className="empty-state">Ask your first question.</div>
            ) : (
              activeSession?.messages.map((message, index) => (
                <div key={index} className={`chat-bubble ${message.role}`}>
                  {message.content}
                </div>
              ))
            )}
          </div>
          <form className="chat-input" onSubmit={handleSubmit}>
            <input
              type="text"
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              placeholder="Ask about findings, methods, or key results..."
            />
            <button className="btn primary" type="submit" disabled={isLoading}>
              {isLoading ? 'Thinking...' : 'Send'}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
};
