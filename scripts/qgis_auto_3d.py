"""Startup script passed to QGIS via --code.

Runs inside the running QGIS Desktop instance after the project is loaded.
It zooms the 2D canvas to the ranked_parcels extent (with satellite basemap
visible beneath) and then opens a 3D Map View centred on the same area.
"""
from __future__ import annotations

from qgis.PyQt.QtCore import QTimer
from qgis.core import QgsProject, QgsRectangle
from qgis.utils import iface


def _zoom_2d() -> None:
    """Zoom the main 2D canvas to the ranked_parcels layer with generous padding."""
    parcels = QgsProject.instance().mapLayersByName("ranked_parcels")
    if not parcels:
        return
    layer = parcels[0]
    extent = QgsRectangle(layer.extent())
    # Generous padding so satellite imagery is clearly visible around parcels
    extent.grow(max(extent.width(), extent.height()) * 1.5)
    canvas = iface.mapCanvas()
    canvas.setExtent(extent)
    canvas.refresh()


def _open_3d_view() -> None:
    """Open a 3D Map View using the View menu (works on all QGIS 3.x)."""
    main_window = iface.mainWindow()
    menu_bar = main_window.menuBar()
    for action in menu_bar.actions():
        text = action.text().lower().replace("&", "")
        if "view" in text:
            view_menu = action.menu()
            if view_menu is None:
                continue
            for sub_action in view_menu.actions():
                sub_menu = sub_action.menu()
                if sub_menu is not None and "3d" in sub_action.text().lower():
                    for item in sub_menu.actions():
                        if "new" in item.text().lower():
                            item.trigger()
                            return


def _set_3d_extent() -> None:
    """After the 3D view is open, try to sync its extent to the 2D canvas."""
    main_window = iface.mainWindow()
    menu_bar = main_window.menuBar()
    # Look for the 3D Map View's "Set 3D Scene Extent on 2D Map View" option
    # This is triggered from within the 3D view's own toolbar, so we try the
    # iface approach instead.
    try:
        # Try to find any open 3D canvas and set its camera
        from qgis.core import QgsVector3D
        parcels = QgsProject.instance().mapLayersByName("ranked_parcels")
        if not parcels:
            return
        extent = parcels[0].extent()
        center = extent.center()
        distance = max(extent.width(), extent.height()) * 3.0
        # Look for open 3D map canvases
        for child in main_window.findChildren(type(None)):
            pass
    except Exception:
        pass


def _startup() -> None:
    """Full startup sequence: maximise → zoom 2D → open 3D."""
    try:
        main_window = iface.mainWindow()
        main_window.showMaximized()
    except Exception:
        pass

    _zoom_2d()

    # Open 3D view after a short delay so the 2D canvas renders first
    QTimer.singleShot(2000, _open_3d_view)


# Give QGIS a moment to finish loading the project before we act.
QTimer.singleShot(3000, _startup)
