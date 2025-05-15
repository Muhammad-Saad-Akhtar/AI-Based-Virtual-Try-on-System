import { useEffect, useState } from 'react';

const GalleryView = ({ onGarmentSelect }) => {
  const [garments, setGarments] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // In a real app, you would fetch the garment list from your server
    // For now, we'll use a hardcoded list of garments in the images folder
    setGarments([
      {
        id: 1,
        name: 'T-Shirt 1',
        path: 'images/tshirt1.jpg',
      },
      {
        id: 2,
        name: 'T-Shirt 2',
        path: 'images/tshirt2.jpg',
      },
      // Add more garments as needed
    ]);
    setLoading(false);
  }, []);

  if (loading) {
    return (
      <div className="text-white text-center">
        Loading garments...
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
      {garments.map(garment => (
        <div
          key={garment.id}
          className="bg-white rounded-lg shadow-xl overflow-hidden cursor-pointer transform hover:scale-105 transition-transform"
          onClick={() => onGarmentSelect(garment)}
        >
          <img
            src={garment.path}
            alt={garment.name}
            className="w-full h-64 object-cover"
          />
          <div className="p-4">
            <h3 className="text-lg font-semibold text-gray-800">{garment.name}</h3>
            <p className="text-sm text-gray-600">Click to try on</p>
          </div>
        </div>
      ))}
    </div>
  );
};

export default GalleryView;
