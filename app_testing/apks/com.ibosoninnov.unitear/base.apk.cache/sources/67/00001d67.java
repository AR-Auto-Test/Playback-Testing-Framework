package com.google.ar.sceneform.ux;

import android.view.View;

/* loaded from: classes.dex */
public class PlaneDiscoveryController {
    private View planeDiscoveryView;

    public PlaneDiscoveryController(View view) {
        this.planeDiscoveryView = view;
    }

    public void hide() {
        View view = this.planeDiscoveryView;
        if (view == null) {
            return;
        }
        view.setVisibility(8);
    }

    public void setInstructionView(View view) {
        this.planeDiscoveryView = view;
    }

    public void show() {
        View view = this.planeDiscoveryView;
        if (view == null) {
            return;
        }
        view.setVisibility(0);
    }
}