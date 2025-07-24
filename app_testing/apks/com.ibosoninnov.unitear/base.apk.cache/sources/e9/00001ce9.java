package com.google.ar.sceneform.rendering;

/* loaded from: classes.dex */
public class MaterialInternalDataGltfImpl extends MaterialInternalData {
    private final com.google.android.filament.Material filamentMaterial;

    public MaterialInternalDataGltfImpl(com.google.android.filament.Material material) {
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
    }
}