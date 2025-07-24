package b.q.b;

import android.animation.LayoutTransition;
import android.content.Context;
import android.content.res.TypedArray;
import android.graphics.Canvas;
import android.os.Bundle;
import android.util.AttributeSet;
import android.view.View;
import android.view.ViewGroup;
import android.view.WindowInsets;
import android.widget.FrameLayout;
import androidx.fragment.app.Fragment;
import com.ibosoninnov.unitear.R;
import java.util.ArrayList;

/* compiled from: FragmentContainerView.java */
/* loaded from: classes.dex */
public final class k extends FrameLayout {

    /* renamed from: b  reason: collision with root package name */
    public ArrayList<View> f2483b;

    /* renamed from: c  reason: collision with root package name */
    public ArrayList<View> f2484c;

    /* renamed from: d  reason: collision with root package name */
    public boolean f2485d;

    public k(Context context) {
        super(context);
        this.f2485d = true;
    }

    public final void a(View view) {
        ArrayList<View> arrayList;
        if (view.getAnimation() != null || ((arrayList = this.f2484c) != null && arrayList.contains(view))) {
            if (this.f2483b == null) {
                this.f2483b = new ArrayList<>();
            }
            this.f2483b.add(view);
        }
    }

    @Override // android.view.ViewGroup
    public void addView(View view, int i, ViewGroup.LayoutParams layoutParams) {
        Object tag = view.getTag(R.id.fragment_container_view_tag);
        if ((tag instanceof Fragment ? (Fragment) tag : null) != null) {
            super.addView(view, i, layoutParams);
            return;
        }
        throw new IllegalStateException("Views added to a FragmentContainerView must be associated with a Fragment. View " + view + " is not associated with a Fragment.");
    }

    @Override // android.view.ViewGroup
    public boolean addViewInLayout(View view, int i, ViewGroup.LayoutParams layoutParams, boolean z) {
        Object tag = view.getTag(R.id.fragment_container_view_tag);
        if ((tag instanceof Fragment ? (Fragment) tag : null) != null) {
            return super.addViewInLayout(view, i, layoutParams, z);
        }
        throw new IllegalStateException("Views added to a FragmentContainerView must be associated with a Fragment. View " + view + " is not associated with a Fragment.");
    }

    @Override // android.view.ViewGroup, android.view.View
    public void dispatchDraw(Canvas canvas) {
        if (this.f2485d && this.f2483b != null) {
            for (int i = 0; i < this.f2483b.size(); i++) {
                super.drawChild(canvas, this.f2483b.get(i), getDrawingTime());
            }
        }
        super.dispatchDraw(canvas);
    }

    @Override // android.view.ViewGroup
    public boolean drawChild(Canvas canvas, View view, long j) {
        ArrayList<View> arrayList;
        if (!this.f2485d || (arrayList = this.f2483b) == null || arrayList.size() <= 0 || !this.f2483b.contains(view)) {
            return super.drawChild(canvas, view, j);
        }
        return false;
    }

    @Override // android.view.ViewGroup
    public void endViewTransition(View view) {
        ArrayList<View> arrayList = this.f2484c;
        if (arrayList != null) {
            arrayList.remove(view);
            ArrayList<View> arrayList2 = this.f2483b;
            if (arrayList2 != null && arrayList2.remove(view)) {
                this.f2485d = true;
            }
        }
        super.endViewTransition(view);
    }

    @Override // android.view.View
    public WindowInsets onApplyWindowInsets(WindowInsets windowInsets) {
        for (int i = 0; i < getChildCount(); i++) {
            getChildAt(i).dispatchApplyWindowInsets(new WindowInsets(windowInsets));
        }
        return windowInsets;
    }

    @Override // android.view.ViewGroup
    public void removeAllViewsInLayout() {
        for (int childCount = getChildCount() - 1; childCount >= 0; childCount--) {
            a(getChildAt(childCount));
        }
        super.removeAllViewsInLayout();
    }

    @Override // android.view.ViewGroup
    public void removeDetachedView(View view, boolean z) {
        if (z) {
            a(view);
        }
        super.removeDetachedView(view, z);
    }

    @Override // android.view.ViewGroup, android.view.ViewManager
    public void removeView(View view) {
        a(view);
        super.removeView(view);
    }

    @Override // android.view.ViewGroup
    public void removeViewAt(int i) {
        a(getChildAt(i));
        super.removeViewAt(i);
    }

    @Override // android.view.ViewGroup
    public void removeViewInLayout(View view) {
        a(view);
        super.removeViewInLayout(view);
    }

    @Override // android.view.ViewGroup
    public void removeViews(int i, int i2) {
        for (int i3 = i; i3 < i + i2; i3++) {
            a(getChildAt(i3));
        }
        super.removeViews(i, i2);
    }

    @Override // android.view.ViewGroup
    public void removeViewsInLayout(int i, int i2) {
        for (int i3 = i; i3 < i + i2; i3++) {
            a(getChildAt(i3));
        }
        super.removeViewsInLayout(i, i2);
    }

    public void setDrawDisappearingViewsLast(boolean z) {
        this.f2485d = z;
    }

    @Override // android.view.ViewGroup
    public void setLayoutTransition(LayoutTransition layoutTransition) {
        throw new UnsupportedOperationException("FragmentContainerView does not support Layout Transitions or animateLayoutChanges=\"true\".");
    }

    @Override // android.view.ViewGroup
    public void startViewTransition(View view) {
        if (view.getParent() == this) {
            if (this.f2484c == null) {
                this.f2484c = new ArrayList<>();
            }
            this.f2484c.add(view);
        }
        super.startViewTransition(view);
    }

    public k(Context context, AttributeSet attributeSet, q qVar) {
        super(context, attributeSet);
        this.f2485d = true;
        String classAttribute = attributeSet.getClassAttribute();
        TypedArray obtainStyledAttributes = context.obtainStyledAttributes(attributeSet, b.q.a.f2396b);
        classAttribute = classAttribute == null ? obtainStyledAttributes.getString(0) : classAttribute;
        String string = obtainStyledAttributes.getString(1);
        obtainStyledAttributes.recycle();
        int id = getId();
        Fragment H = qVar.H(id);
        if (classAttribute == null || H != null) {
            return;
        }
        if (id <= 0) {
            throw new IllegalStateException(c.b.a.a.a.r("FragmentContainerView must have an android:id to add Fragment ", classAttribute, string != null ? c.b.a.a.a.q(" with tag ", string) : ""));
        }
        Fragment a2 = qVar.L().a(context.getClassLoader(), classAttribute);
        a2.onInflate(context, attributeSet, (Bundle) null);
        a aVar = new a(qVar);
        aVar.p = true;
        a2.mContainer = this;
        aVar.d(getId(), a2, string, 1);
        if (!aVar.f2547g) {
            aVar.f2548h = false;
            aVar.q.D(aVar, true);
            return;
        }
        throw new IllegalStateException("This transaction is already being added to the back stack");
    }
}