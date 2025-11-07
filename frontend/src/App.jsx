// src/App.jsx
import { useEffect, useState } from "react";
import NavBar from "./components/NavBar";
import HomePage from "./pages/HomePage";
import ChatBot from "./components/ChatBot";
import FAQPage from "./pages/FAQPage";
import IssuePage from "./pages/IssuePage";
import TechnologyPage from "./pages/TechnologyPage";
import Footer from "./components/Footer";
import { Routes, Route, useLocation } from "react-router-dom";

function App() {
  useEffect(() => {}, []);
  const [currentPage, setCurrentPage] = useState("Home");
  const location = useLocation();
  const isChatPage = location.pathname.endsWith("/chat") || location.pathname.endsWith("/chat/");

  return (
    <div className="flex flex-col h-screen overflow-hidden">
      <NavBar />
      <main className="flex-1 flex overflow-hidden">
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="chat" element={<ChatBot />} />
          <Route path="issue" element={<IssuePage />} />
          <Route path="faq" element={<FAQPage />} />
          <Route path="technology" element={<TechnologyPage />} />
        </Routes>
      </main>
      {!isChatPage && <Footer />}
    </div>
  );
}

export default App;
