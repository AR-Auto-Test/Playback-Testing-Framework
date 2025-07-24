package com.google.ar.sceneform.rendering;

import com.google.ar.sceneform.utilities.AndroidPreconditions;

/* loaded from: classes.dex */
public class MaterialInternalDataImpl extends MaterialInternalData {
    private com.google.android.filament.Material filamentMaterial;

    public MaterialInternalDataImpl(com.google.android.filament.Material material) {
        this.filamentMaterial = material;
    }

    @Override // com.google.ar.sceneform.rendering.MaterialInternalData
    public com.google.android.filament.Material getFilamentMaterial() {
        com.google.android.filament.Material material = this.filamentMaterial;
        if (material != null) {
            return material;
        }
        throw new IllegalStateException("Filament Material is null.");
    }

    @Override // com.google.ar.sceneform.resources.SharedReference
    public void onDispose() {
        AndroidPreconditions.checkUiThread();
        IEngine engine = EngineInstance.getEngine();
        com.google.android.filament.Material material = this.filamentMaterial;
        this.filamentMaterial = null;
        if (material == null || engine == null || !engine.isValid()) {
            return;
        }
        engine.destroyMaterial(material);
    }
}