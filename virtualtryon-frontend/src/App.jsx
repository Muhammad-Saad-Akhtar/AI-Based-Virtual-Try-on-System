import { useState } from 'react';
import './assets/styles/App.css';
import GalleryView from './components/GalleryView';
import VirtualTryOn from './components/VirtualTryOn';

function App() {
  const [activeView, setActiveView] = useState('gallery'); // 'gallery' or 'tryon'
  const [selectedGarment, setSelectedGarment] = useState(null);

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-900 via-purple-800 to-indigo-800">
      <header className="py-8 text-center">
        <h1 className="text-6xl font-bold text-white mb-2 animate-fade-in">
          Virtual Try-On
        </h1>
        <p className="text-purple-200 text-xl animate-fade-in-delayed">
          Try on garments virtually using AI
        </p>
      </header>

      <main className="container mx-auto px-4">
        {activeView === 'gallery' ? (
          <GalleryView 
            onGarmentSelect={(garment) => {
              setSelectedGarment(garment);
              setActiveView('tryon');
            }}
          />
        ) : (
          <VirtualTryOn
            selectedGarment={selectedGarment}
            onBack={() => setActiveView('gallery')}
          />
        )}
      </main>
    </div>
  );
}

export default App;
