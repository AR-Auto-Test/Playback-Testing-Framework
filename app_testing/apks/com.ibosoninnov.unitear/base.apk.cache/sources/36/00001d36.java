package com.google.ar.sceneform.rendering;

import com.google.ar.sceneform.resources.SharedReference;
import com.google.ar.sceneform.utilities.AndroidPreconditions;

/* loaded from: classes.dex */
public class ViewRenderableInternalData extends SharedReference {
    private final RenderViewToExternalTexture renderView;

    public ViewRenderableInternalData(RenderViewToExternalTexture renderViewToExternalTexture) {
        this.renderView = renderViewToExternalTexture;
    }

    public RenderViewToExternalTexture getRenderView() {
        return this.renderView;
    }

    @Override // com.google.ar.sceneform.resources.SharedReference
    public void onDispose() {
        AndroidPreconditions.checkUiThread();
        this.renderView.releaseResources();
    }
}