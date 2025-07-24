package androidx.appcompat.widget;

import android.content.Context;
import android.content.res.TypedArray;
import android.graphics.drawable.Drawable;
import android.util.AttributeSet;
import android.view.ActionMode;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewGroup;
import android.widget.FrameLayout;
import b.b.h.b;
import b.b.h.q0;
import b.j.j.q;
import com.google.ar.core.ImageMetadata;
import com.ibosoninnov.unitear.R;
import java.util.concurrent.atomic.AtomicInteger;
import org.opencv.calib3d.Calib3d;

/* loaded from: classes.dex */
public class ActionBarContainer extends FrameLayout {

    /* renamed from: b  reason: collision with root package name */
    public boolean f95b;

    /* renamed from: c  reason: collision with root package name */
    public View f96c;

    /* renamed from: d  reason: collision with root package name */
    public View f97d;

    /* renamed from: e  reason: collision with root package name */
    public View f98e;

    /* renamed from: f  reason: collision with root package name */
    public Drawable f99f;

    /* renamed from: g  reason: collision with root package name */
    public Drawable f100g;

    /* renamed from: h  reason: collision with root package name */
    public Drawable f101h;
    public boolean i;
    public boolean j;
    public int k;

    public ActionBarContainer(Context context, AttributeSet attributeSet) {
        super(context, attributeSet);
        b bVar = new b(this);
        AtomicInteger atomicInteger = q.f2214a;
        setBackground(bVar);
        TypedArray obtainStyledAttributes = context.obtainStyledAttributes(attributeSet, b.b.b.f541a);
        boolean z = false;
        this.f99f = obtainStyledAttributes.getDrawable(0);
        this.f100g = obtainStyledAttributes.getDrawable(2);
        this.k = obtainStyledAttributes.getDimensionPixelSize(13, -1);
        if (getId() == R.id.split_action_bar) {
            this.i = true;
            this.f101h = obtainStyledAttributes.getDrawable(1);
        }
        obtainStyledAttributes.recycle();
        if (!this.i ? !(this.f99f != null || this.f100g != null) : this.f101h == null) {
            z = true;
        }
        setWillNotDraw(z);
    }

    public final int a(View view) {
        FrameLayout.LayoutParams layoutParams = (FrameLayout.LayoutParams) view.getLayoutParams();
        return view.getMeasuredHeight() + layoutParams.topMargin + layoutParams.bottomMargin;
    }

    public final boolean b(View view) {
        return view == null || view.getVisibility() == 8 || view.getMeasuredHeight() == 0;
    }

    @Override // android.view.ViewGroup, android.view.View
    public void drawableStateChanged() {
        super.drawableStateChanged();
        Drawable drawable = this.f99f;
        if (drawable != null && drawable.isStateful()) {
            this.f99f.setState(getDrawableState());
        }
        Drawable drawable2 = this.f100g;
        if (drawable2 != null && drawable2.isStateful()) {
            this.f100g.setState(getDrawableState());
        }
        Drawable drawable3 = this.f101h;
        if (drawable3 == null || !drawable3.isStateful()) {
            return;
        }
        this.f101h.setState(getDrawableState());
    }

    public View getTabContainer() {
        return this.f96c;
    }

    @Override // android.view.ViewGroup, android.view.View
    public void jumpDrawablesToCurrentState() {
        super.jumpDrawablesToCurrentState();
        Drawable drawable = this.f99f;
        if (drawable != null) {
            drawable.jumpToCurrentState();
        }
        Drawable drawable2 = this.f100g;
        if (drawable2 != null) {
            drawable2.jumpToCurrentState();
        }
        Drawable drawable3 = this.f101h;
        if (drawable3 != null) {
            drawable3.jumpToCurrentState();
        }
    }

    @Override // android.view.View
    public void onFinishInflate() {
        super.onFinishInflate();
        this.f97d = findViewById(R.id.action_bar);
        this.f98e = findViewById(R.id.action_context_bar);
    }

    @Override // android.view.View
    public boolean onHoverEvent(MotionEvent motionEvent) {
        super.onHoverEvent(motionEvent);
        return true;
    }

    @Override // android.view.ViewGroup
    public boolean onInterceptTouchEvent(MotionEvent motionEvent) {
        return this.f95b || super.onInterceptTouchEvent(motionEvent);
    }

    @Override // android.widget.FrameLayout, android.view.ViewGroup, android.view.View
    public void onLayout(boolean z, int i, int i2, int i3, int i4) {
        Drawable drawable;
        super.onLayout(z, i, i2, i3, i4);
        View view = this.f96c;
        boolean z2 = true;
        boolean z3 = false;
        boolean z4 = (view == null || view.getVisibility() == 8) ? false : true;
        if (view != null && view.getVisibility() != 8) {
            int measuredHeight = getMeasuredHeight();
            int i5 = ((FrameLayout.LayoutParams) view.getLayoutParams()).bottomMargin;
            view.layout(i, (measuredHeight - view.getMeasuredHeight()) - i5, i3, measuredHeight - i5);
        }
        if (this.i) {
            Drawable drawable2 = this.f101h;
            if (drawable2 != null) {
                drawable2.setBounds(0, 0, getMeasuredWidth(), getMeasuredHeight());
            }
            z2 = z3;
        } else {
            if (this.f99f != null) {
                if (this.f97d.getVisibility() == 0) {
                    this.f99f.setBounds(this.f97d.getLeft(), this.f97d.getTop(), this.f97d.getRight(), this.f97d.getBottom());
                } else {
                    View view2 = this.f98e;
                    if (view2 != null && view2.getVisibility() == 0) {
                        this.f99f.setBounds(this.f98e.getLeft(), this.f98e.getTop(), this.f98e.getRight(), this.f98e.getBottom());
                    } else {
                        this.f99f.setBounds(0, 0, 0, 0);
                    }
                }
                z3 = true;
            }
            this.j = z4;
            if (z4 && (drawable = this.f100g) != null) {
                drawable.setBounds(view.getLeft(), view.getTop(), view.getRight(), view.getBottom());
            }
            z2 = z3;
        }
        if (z2) {
            invalidate();
        }
    }

    @Override // android.widget.FrameLayout, android.view.View
    public void onMeasure(int i, int i2) {
        int a2;
        int i3;
        if (this.f97d == null && View.MeasureSpec.getMode(i2) == Integer.MIN_VALUE && (i3 = this.k) >= 0) {
            i2 = View.MeasureSpec.makeMeasureSpec(Math.min(i3, View.MeasureSpec.getSize(i2)), Integer.MIN_VALUE);
        }
        super.onMeasure(i, i2);
        if (this.f97d == null) {
            return;
        }
        int mode = View.MeasureSpec.getMode(i2);
        View view = this.f96c;
        if (view == null || view.getVisibility() == 8 || mode == 1073741824) {
            return;
        }
        if (!b(this.f97d)) {
            a2 = a(this.f97d);
        } else {
            a2 = !b(this.f98e) ? a(this.f98e) : 0;
        }
        setMeasuredDimension(getMeasuredWidth(), Math.min(a(this.f96c) + a2, mode == Integer.MIN_VALUE ? View.MeasureSpec.getSize(i2) : Integer.MAX_VALUE));
    }

    @Override // android.view.View
    public boolean onTouchEvent(MotionEvent motionEvent) {
        super.onTouchEvent(motionEvent);
        return true;
    }

    public void setPrimaryBackground(Drawable drawable) {
        Drawable drawable2 = this.f99f;
        if (drawable2 != null) {
            drawable2.setCallback(null);
            unscheduleDrawable(this.f99f);
        }
        this.f99f = drawable;
        if (drawable != null) {
            drawable.setCallback(this);
            View view = this.f97d;
            if (view != null) {
                this.f99f.setBounds(view.getLeft(), this.f97d.getTop(), this.f97d.getRight(), this.f97d.getBottom());
            }
        }
        boolean z = true;
        if (!this.i ? this.f99f != null || this.f100g != null : this.f101h != null) {
            z = false;
        }
        setWillNotDraw(z);
        invalidate();
        invalidateOutline();
    }

    public void setSplitBackground(Drawable drawable) {
        Drawable drawable2;
        Drawable drawable3 = this.f101h;
        if (drawable3 != null) {
            drawable3.setCallback(null);
            unscheduleDrawable(this.f101h);
        }
        this.f101h = drawable;
        boolean z = false;
        if (drawable != null) {
            drawable.setCallback(this);
            if (this.i && (drawable2 = this.f101h) != null) {
                drawable2.setBounds(0, 0, getMeasuredWidth(), getMeasuredHeight());
            }
        }
        if (!this.i ? !(this.f99f != null || this.f100g != null) : this.f101h == null) {
            z = true;
        }
        setWillNotDraw(z);
        invalidate();
        invalidateOutline();
    }

    public void setStackedBackground(Drawable drawable) {
        Drawable drawable2;
        Drawable drawable3 = this.f100g;
        if (drawable3 != null) {
            drawable3.setCallback(null);
            unscheduleDrawable(this.f100g);
        }
        this.f100g = drawable;
        if (drawable != null) {
            drawable.setCallback(this);
            if (this.j && (drawable2 = this.f100g) != null) {
                drawable2.setBounds(this.f96c.getLeft(), this.f96c.getTop(), this.f96c.getRight(), this.f96c.getBottom());
            }
        }
        boolean z = true;
        if (!this.i ? this.f99f != null || this.f100g != null : this.f101h != null) {
            z = false;
        }
        setWillNotDraw(z);
        invalidate();
        invalidateOutline();
    }

    public void setTabContainer(q0 q0Var) {
        View view = this.f96c;
        if (view != null) {
            removeView(view);
        }
        this.f96c = q0Var;
        if (q0Var != null) {
            addView(q0Var);
            ViewGroup.LayoutParams layoutParams = q0Var.getLayoutParams();
            layoutParams.width = -1;
            layoutParams.height = -2;
            q0Var.setAllowCollapse(false);
        }
    }

    public void setTransitioning(boolean z) {
        this.f95b = z;
        setDescendantFocusability(z ? ImageMetadata.HOT_PIXEL_MODE : Calib3d.CALIB_TILTED_MODEL);
    }

    @Override // android.view.View
    public void setVisibility(int i) {
        super.setVisibility(i);
        boolean z = i == 0;
        Drawable drawable = this.f99f;
        if (drawable != null) {
            drawable.setVisible(z, false);
        }
        Drawable drawable2 = this.f100g;
        if (drawable2 != null) {
            drawable2.setVisible(z, false);
        }
        Drawable drawable3 = this.f101h;
        if (drawable3 != null) {
            drawable3.setVisible(z, false);
        }
    }

    @Override // android.view.ViewGroup, android.view.ViewParent
    public ActionMode startActionModeForChild(View view, ActionMode.Callback callback) {
        return null;
    }

    @Override // android.view.ViewGroup, android.view.ViewParent
    public ActionMode startActionModeForChild(View view, ActionMode.Callback callback, int i) {
        if (i != 0) {
            return super.startActionModeForChild(view, callback, i);
        }
        return null;
    }

    @Override // android.view.View
    public boolean verifyDrawable(Drawable drawable) {
        return (drawable == this.f99f && !this.i) || (drawable == this.f100g && this.j) || ((drawable == this.f101h && this.i) || super.verifyDrawable(drawable));
    }
}