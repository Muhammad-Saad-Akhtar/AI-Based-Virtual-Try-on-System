const express = require('express');
const fs = require('fs');
const path = require('path');
const cors = require('cors'); // Allow React frontend to connect

const app = express();
const PORT = 9000;

const IMAGES_DIR = path.join(__dirname, 'assets', 'Images');
app.use('/images', express.static(IMAGES_DIR));

app.use(cors());

app.get('/api/images', (req, res) => {
  fs.readdir(IMAGES_DIR, (err, files) => {
    if (err) {
      console.error('Error reading image directory:', err);
      return res.status(500).json({ error: 'Failed to read image folder' });
    }

    const imageFiles = files.filter(file => /\.(jpg|jpeg|png|gif)$/i.test(file));
    res.json(imageFiles);
  });
});

app.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}`);
});
