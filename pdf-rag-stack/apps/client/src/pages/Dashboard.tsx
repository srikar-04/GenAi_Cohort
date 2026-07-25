import { useListDocumentsQuery } from '../features/documents/documentsApi';

export const Dashboard = () => {
  const { data } = useListDocumentsQuery();
  const documents = data?.documents ?? [];
  const processing = documents.filter((doc) => doc.status === 'PROCESSING').length;
  const ready = documents.filter((doc) => doc.status === 'READY').length;

  return (
    <div className="page">
      <header className="page-header">
        <h2>Dashboard</h2>
        <p>Track ingestion and jump into chats with your most recent PDFs.</p>
      </header>
      <div className="card-grid">
        <div className="stat-card">
          <h3>{documents.length}</h3>
          <span>Total documents</span>
        </div>
        <div className="stat-card">
          <h3>{processing}</h3>
          <span>Processing</span>
        </div>
        <div className="stat-card">
          <h3>{ready}</h3>
          <span>Ready to query</span>
        </div>
      </div>
      <div className="note-card">
        <h4>Tip</h4>
        <p>
          For research papers, choose the <strong>Research</strong> document type to apply
          section-aware chunking and reference trimming.
        </p>
      </div>
    </div>
  );
};
