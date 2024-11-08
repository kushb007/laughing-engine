<head>
 <meta charset="utf-8">
 <script src="https://ajax.googleapis.com/ajax/libs/cesiumjs/1.105/Build/Cesium/Cesium.js"></script>
 <link href="https://ajax.googleapis.com/ajax/libs/cesiumjs/1.105/Build/Cesium/Widgets/widgets.css" rel="stylesheet">
</head>

<body>
  <div id="cesiumContainer"></div>
  <script>
    const viewer = new Cesium.Viewer('cesiumContainer', {
      imageryProvider: false,
      baseLayerPicker: false,
      requestRenderMode: true,
    });

    const tileset = viewer.scene.primitives.add(new Cesium.Cesium3DTileset({
      url: "https://tile.googleapis.com/v1/3dtiles/root.json?key=YOUR_API_KEY",
      showCreditsOnScreen: true,
    }));

    viewer.scene.globe.show = false;
  </script>
</body>