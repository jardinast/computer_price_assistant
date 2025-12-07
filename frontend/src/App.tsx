import { Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import Home from './pages/Home';
import MarketOverview from './pages/MarketOverview';
import PricePredictor from './pages/PricePredictor';
// @ts-ignore
import ChatAdvisor from './pages/ChatAdvisor';
import SimilarLaptops from './pages/SimilarLaptops';

function App() {
  return (
    <Routes>
      <Route path="/" element={<Layout />}>
        <Route index element={<Home />} />
        <Route path="market" element={<MarketOverview />} />
        <Route path="predict" element={<PricePredictor />} />
        <Route path="chat" element={<ChatAdvisor />} />
        <Route path="similar" element={<SimilarLaptops />} />
      </Route>
    </Routes>
  );
}

export default App;
