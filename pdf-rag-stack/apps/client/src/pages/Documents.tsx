import { UploadPanel } from '../components/documents/UploadPanel';
import { DocumentList } from '../components/documents/DocumentList';
import { useListDocumentsQuery } from '../features/documents/documentsApi';

export const Documents = () => {
  const { data, isLoading } = useListDocumentsQuery();
  const documents = data?.documents ?? [];

  return (
    <div className="page">
      <header className="page-header">
        <h2>Documents</h2>
        <p>Upload PDFs and track ingestion progress.</p>
      </header>
      <UploadPanel />
      {isLoading ? <div className="loading-inline">Loading documents...</div> : null}
      <DocumentList documents={documents} />
    </div>
  );
};
