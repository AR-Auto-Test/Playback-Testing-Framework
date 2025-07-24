package com.google.ar.sceneform.rendering;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Picture;
import android.graphics.PorterDuff;
import android.view.Surface;
import android.view.View;
import android.widget.LinearLayout;
import com.google.ar.sceneform.utilities.Preconditions;
import java.util.ArrayList;
import java.util.Iterator;

/* loaded from: classes.dex */
public class RenderViewToExternalTexture extends LinearLayout {
    private final ExternalTexture externalTexture;
    private boolean hasDrawnToSurfaceTexture;
    private final ArrayList<OnViewSizeChangedListener> onViewSizeChangedListeners;
    private final Picture picture;
    private final View view;
    private ViewAttachmentManager viewAttachmentManager;

    /* loaded from: classes.dex */
    public interface OnViewSizeChangedListener {
        void onViewSizeChanged(int i, int i2);
    }

    public RenderViewToExternalTexture(Context context, View view) {
        super(context);
        this.picture = new Picture();
        this.hasDrawnToSurfaceTexture = false;
        this.onViewSizeChangedListeners = new ArrayList<>();
        Preconditions.checkNotNull(view, "Parameter \"view\" was null.");
        this.externalTexture = new ExternalTexture();
        this.view = view;
        addView(view);
    }

    public void addOnViewSizeChangedListener(OnViewSizeChangedListener onViewSizeChangedListener) {
        if (this.onViewSizeChangedListeners.contains(onViewSizeChangedListener)) {
            return;
        }
        this.onViewSizeChangedListeners.add(onViewSizeChangedListener);
    }

    public void attachView(ViewAttachmentManager viewAttachmentManager) {
        ViewAttachmentManager viewAttachmentManager2 = this.viewAttachmentManager;
        if (viewAttachmentManager2 != null) {
            if (viewAttachmentManager2 != viewAttachmentManager) {
                throw new IllegalStateException("Cannot use the same ViewRenderable with multiple SceneViews.");
            }
            return;
        }
        this.viewAttachmentManager = viewAttachmentManager;
        viewAttachmentManager.addView(this);
    }

    public void detachView() {
        ViewAttachmentManager viewAttachmentManager = this.viewAttachmentManager;
        if (viewAttachmentManager != null) {
            viewAttachmentManager.removeView(this);
            this.viewAttachmentManager = null;
        }
    }

    @Override // android.view.ViewGroup, android.view.View
    public void dispatchDraw(Canvas canvas) {
        Surface surface = this.externalTexture.getSurface();
        if (surface.isValid()) {
            if (this.view.isDirty()) {
                Canvas beginRecording = this.picture.beginRecording(this.view.getWidth(), this.view.getHeight());
                beginRecording.drawColor(0, PorterDuff.Mode.CLEAR);
                super.dispatchDraw(beginRecording);
                this.picture.endRecording();
                Canvas lockCanvas = surface.lockCanvas(null);
                this.picture.draw(lockCanvas);
                surface.unlockCanvasAndPost(lockCanvas);
                this.hasDrawnToSurfaceTexture = true;
            }
            invalidate();
        }
    }

    public ExternalTexture getExternalTexture() {
        return this.externalTexture;
    }

    public boolean hasDrawnToSurfaceTexture() {
        return this.hasDrawnToSurfaceTexture;
    }

    @Override // android.view.ViewGroup, android.view.View
    public void onAttachedToWindow() {
        super.onAttachedToWindow();
    }

    @Override // android.widget.LinearLayout, android.view.ViewGroup, android.view.View
    public void onLayout(boolean z, int i, int i2, int i3, int i4) {
        super.onLayout(z, i, i2, i3, i4);
        this.externalTexture.getSurfaceTexture().setDefaultBufferSize(this.view.getWidth(), this.view.getHeight());
    }

    @Override // android.view.View
    public void onSizeChanged(int i, int i2, int i3, int i4) {
        Iterator<OnViewSizeChangedListener> it = this.onViewSizeChangedListeners.iterator();
        while (it.hasNext()) {
            it.next().onViewSizeChanged(i, i2);
        }
    }

    public void releaseResources() {
        detachView();
    }

    public void removeOnViewSizeChangedListener(OnViewSizeChangedListener onViewSizeChangedListener) {
        this.onViewSizeChangedListeners.remove(onViewSizeChangedListener);
    }
}