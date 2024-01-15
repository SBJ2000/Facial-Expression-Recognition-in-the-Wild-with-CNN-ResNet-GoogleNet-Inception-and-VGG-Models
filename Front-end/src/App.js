import React, { useState, useEffect } from "react";
import Preloader from "../src/Components/Pre";
import Navbar from "./Components/Navbar";
import Home from "./Components/Home/Home";
import Resnet from "./Components/Resnet/Resnet";
import CNN from "./Components/CNN/CNN";
import Inception from "./Components/Inception/Inception";
import GoogleNet from "./Components/GoogleNet/GoogleNet";
import Vgg from "./Components/Vgg/Vgg";

import {
  BrowserRouter as Router,
  Route,
  Routes,
  Navigate
} from "react-router-dom";
import ScrollToTop from "./Components/ScrollToTop";
import "./style.css";
import "./App.css";
import "bootstrap/dist/css/bootstrap.min.css";

function App() {
  const [load, upadateLoad] = useState(true);

  useEffect(() => {
    const timer = setTimeout(() => {
      upadateLoad(false);
    }, 1200);

    return () => clearTimeout(timer);
  }, []);

  return (
    <Router>
      <Preloader load={load} />
      <div className="App" id={load ? "no-scroll" : "scroll"}>
        <Navbar />
        <ScrollToTop />
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/Resnet" element={<Resnet />} />
          <Route path="/CNN" element={<CNN />} />
          <Route path="/Inception" element={<Inception />} />
          <Route path="/GoogleNet" element={<GoogleNet />} />
          <Route path="/Vgg" element={<Vgg />} />
          <Route path="*" element={<Navigate to="/"/>} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;