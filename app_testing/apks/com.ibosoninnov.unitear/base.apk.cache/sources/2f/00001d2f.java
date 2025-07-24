package com.google.ar.sceneform.rendering;

import android.content.Context;
import android.view.View;
import android.view.ViewGroup;
import android.view.ViewParent;
import android.view.WindowManager;
import android.widget.FrameLayout;
import com.google.ar.sceneform.rendering.ViewAttachmentManager;

/* loaded from: classes.dex */
public class ViewAttachmentManager {
    private static final String VIEW_RENDERABLE_WINDOW = "ViewRenderableWindow";
    private final FrameLayout frameLayout;
    private final View ownerView;
    private final WindowManager windowManager;
    private final WindowManager.LayoutParams windowLayoutParams = createWindowLayoutParams();
    private final ViewGroup.LayoutParams viewLayoutParams = createViewLayoutParams();

    public ViewAttachmentManager(Context context, View view) {
        this.ownerView = view;
        this.windowManager = (WindowManager) context.getSystemService("window");
        this.frameLayout = new FrameLayout(context);
    }

    private static ViewGroup.LayoutParams createViewLayoutParams() {
        return new ViewGroup.LayoutParams(-2, -2);
    }

    private static WindowManager.LayoutParams createWindowLayoutParams() {
        WindowManager.LayoutParams layoutParams = new WindowManager.LayoutParams(-2, -2, 1000, 16777752, -3);
        layoutParams.setTitle(VIEW_RENDERABLE_WINDOW);
        return layoutParams;
    }

    public /* synthetic */ void a() {
        if (this.frameLayout.getParent() == null && this.ownerView.isAttachedToWindow()) {
            this.windowManager.addView(this.frameLayout, this.windowLayoutParams);
        }
    }

    public void addView(View view) {
        ViewParent parent = view.getParent();
        FrameLayout frameLayout = this.frameLayout;
        if (parent == frameLayout) {
            return;
        }
        frameLayout.addView(view, this.viewLayoutParams);
    }

    public FrameLayout getFrameLayout() {
        return this.frameLayout;
    }

    public void onPause() {
        if (this.frameLayout.getParent() != null) {
            this.windowManager.removeView(this.frameLayout);
        }
    }

    public void onResume() {
        this.ownerView.post(new Runnable() { // from class: c.d.b.a.q.k0
            @Override // java.lang.Runnable
            public final void run() {
                ViewAttachmentManager.this.a();
            }
        });
    }

    public void removeView(View view) {
        ViewParent parent = view.getParent();
        FrameLayout frameLayout = this.frameLayout;
        if (parent != frameLayout) {
            return;
        }
        frameLayout.removeView(view);
    }
}