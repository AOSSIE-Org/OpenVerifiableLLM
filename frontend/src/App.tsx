import { HashRouter, Route, Routes } from 'react-router-dom';
import { AppShell } from './layout/AppShell';
import { EvidenceDetailPage } from './pages/EvidenceDetailPage';
import { EvidenceListPage } from './pages/EvidenceListPage';
import { OverviewPage } from './pages/OverviewPage';
import { ReleasesPage } from './pages/ReleasesPage';
import { TryPage } from './pages/TryPage';
import { VerifyPage } from './pages/VerifyPage';
import { SnapshotProvider } from './snapshot';

export default function App() {
  return (
    <HashRouter>
      <SnapshotProvider>
        <Routes>
          <Route element={<AppShell />}>
            <Route path="/" element={<OverviewPage />} />
            <Route path="/evidence" element={<EvidenceListPage />} />
            <Route path="/evidence/:id" element={<EvidenceDetailPage />} />
            <Route path="/releases" element={<ReleasesPage />} />
            <Route path="/verify" element={<VerifyPage />} />
            <Route path="/try" element={<TryPage />} />
          </Route>
        </Routes>
      </SnapshotProvider>
    </HashRouter>
  );
}
