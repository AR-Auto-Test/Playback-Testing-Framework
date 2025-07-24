package androidx.appcompat.widget;

import android.content.Context;
import android.graphics.Rect;
import android.util.AttributeSet;
import android.util.DisplayMetrics;
import android.util.TypedValue;
import android.view.View;
import android.widget.FrameLayout;
import b.b.c.k;
import b.b.c.m;
import b.b.g.i.g;
import b.b.h.c0;
import java.util.Objects;

/* loaded from: classes.dex */
public class ContentFrameLayout extends FrameLayout {

    /* renamed from: b  reason: collision with root package name */
    public TypedValue f130b;

    /* renamed from: c  reason: collision with root package name */
    public TypedValue f131c;

    /* renamed from: d  reason: collision with root package name */
    public TypedValue f132d;

    /* renamed from: e  reason: collision with root package name */
    public TypedValue f133e;

    /* renamed from: f  reason: collision with root package name */
    public TypedValue f134f;

    /* renamed from: g  reason: collision with root package name */
    public TypedValue f135g;

    /* renamed from: h  reason: collision with root package name */
    public final Rect f136h;
    public a i;

    /* loaded from: classes.dex */
    public interface a {
    }

    public ContentFrameLayout(Context context, AttributeSet attributeSet) {
        super(context, attributeSet, 0);
        this.f136h = new Rect();
    }

    public TypedValue getFixedHeightMajor() {
        if (this.f134f == null) {
            this.f134f = new TypedValue();
        }
        return this.f134f;
    }

    public TypedValue getFixedHeightMinor() {
        if (this.f135g == null) {
            this.f135g = new TypedValue();
        }
        return this.f135g;
    }

    public TypedValue getFixedWidthMajor() {
        if (this.f132d == null) {
            this.f132d = new TypedValue();
        }
        return this.f132d;
    }

    public TypedValue getFixedWidthMinor() {
        if (this.f133e == null) {
            this.f133e = new TypedValue();
        }
        return this.f133e;
    }

    public TypedValue getMinWidthMajor() {
        if (this.f130b == null) {
            this.f130b = new TypedValue();
        }
        return this.f130b;
    }

    public TypedValue getMinWidthMinor() {
        if (this.f131c == null) {
            this.f131c = new TypedValue();
        }
        return this.f131c;
    }

    @Override // android.view.ViewGroup, android.view.View
    public void onAttachedToWindow() {
        super.onAttachedToWindow();
        a aVar = this.i;
        if (aVar != null) {
            Objects.requireNonNull((m) aVar);
        }
    }

    @Override // android.view.ViewGroup, android.view.View
    public void onDetachedFromWindow() {
        super.onDetachedFromWindow();
        a aVar = this.i;
        if (aVar != null) {
            k kVar = ((m) aVar).f595a;
            c0 c0Var = kVar.p;
            if (c0Var != null) {
                c0Var.i();
            }
            if (kVar.u != null) {
                kVar.j.getDecorView().removeCallbacks(kVar.v);
                if (kVar.u.isShowing()) {
                    try {
                        kVar.u.dismiss();
                    } catch (IllegalArgumentException unused) {
                    }
                }
                kVar.u = null;
            }
            kVar.G();
            g gVar = kVar.L(0).f592h;
            if (gVar != null) {
                gVar.close();
            }
        }
    }

    /* JADX WARN: Removed duplicated region for block: B:21:0x004a  */
    /* JADX WARN: Removed duplicated region for block: B:24:0x0063  */
    /* JADX WARN: Removed duplicated region for block: B:37:0x0086  */
    /* JADX WARN: Removed duplicated region for block: B:42:0x00ab  */
    /* JADX WARN: Removed duplicated region for block: B:43:0x00ae  */
    /* JADX WARN: Removed duplicated region for block: B:48:0x00b8  */
    /* JADX WARN: Removed duplicated region for block: B:50:0x00be  */
    /* JADX WARN: Removed duplicated region for block: B:54:0x00cc  */
    /* JADX WARN: Removed duplicated region for block: B:56:0x00d6  */
    /* JADX WARN: Removed duplicated region for block: B:59:0x00de  */
    /* JADX WARN: Removed duplicated region for block: B:61:? A[RETURN, SYNTHETIC] */
    @Override // android.widget.FrameLayout, android.view.View
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public void onMeasure(int i, int i2) {
        boolean z;
        int measuredWidth;
        TypedValue typedValue;
        int i3;
        int i4;
        float fraction;
        int i5;
        int i6;
        float fraction2;
        int i7;
        int i8;
        float fraction3;
        DisplayMetrics displayMetrics = getContext().getResources().getDisplayMetrics();
        boolean z2 = true;
        boolean z3 = displayMetrics.widthPixels < displayMetrics.heightPixels;
        int mode = View.MeasureSpec.getMode(i);
        int mode2 = View.MeasureSpec.getMode(i2);
        if (mode == Integer.MIN_VALUE) {
            TypedValue typedValue2 = z3 ? this.f133e : this.f132d;
            if (typedValue2 != null && (i7 = typedValue2.type) != 0) {
                if (i7 == 5) {
                    fraction3 = typedValue2.getDimension(displayMetrics);
                } else if (i7 == 6) {
                    int i9 = displayMetrics.widthPixels;
                    fraction3 = typedValue2.getFraction(i9, i9);
                } else {
                    i8 = 0;
                    if (i8 > 0) {
                        Rect rect = this.f136h;
                        i = View.MeasureSpec.makeMeasureSpec(Math.min(i8 - (rect.left + rect.right), View.MeasureSpec.getSize(i)), 1073741824);
                        z = true;
                        if (mode2 == Integer.MIN_VALUE) {
                            TypedValue typedValue3 = z3 ? this.f134f : this.f135g;
                            if (typedValue3 != null && (i5 = typedValue3.type) != 0) {
                                if (i5 == 5) {
                                    fraction2 = typedValue3.getDimension(displayMetrics);
                                } else if (i5 == 6) {
                                    int i10 = displayMetrics.heightPixels;
                                    fraction2 = typedValue3.getFraction(i10, i10);
                                } else {
                                    i6 = 0;
                                    if (i6 > 0) {
                                        Rect rect2 = this.f136h;
                                        i2 = View.MeasureSpec.makeMeasureSpec(Math.min(i6 - (rect2.top + rect2.bottom), View.MeasureSpec.getSize(i2)), 1073741824);
                                    }
                                }
                                i6 = (int) fraction2;
                                if (i6 > 0) {
                                }
                            }
                        }
                        super.onMeasure(i, i2);
                        measuredWidth = getMeasuredWidth();
                        int makeMeasureSpec = View.MeasureSpec.makeMeasureSpec(measuredWidth, 1073741824);
                        if (!z && mode == Integer.MIN_VALUE) {
                            typedValue = !z3 ? this.f131c : this.f130b;
                            if (typedValue != null && (i3 = typedValue.type) != 0) {
                                if (i3 != 5) {
                                    fraction = typedValue.getDimension(displayMetrics);
                                } else if (i3 == 6) {
                                    int i11 = displayMetrics.widthPixels;
                                    fraction = typedValue.getFraction(i11, i11);
                                } else {
                                    i4 = 0;
                                    if (i4 > 0) {
                                        Rect rect3 = this.f136h;
                                        i4 -= rect3.left + rect3.right;
                                    }
                                    if (measuredWidth < i4) {
                                        makeMeasureSpec = View.MeasureSpec.makeMeasureSpec(i4, 1073741824);
                                        if (z2) {
                                            super.onMeasure(makeMeasureSpec, i2);
                                            return;
                                        }
                                        return;
                                    }
                                }
                                i4 = (int) fraction;
                                if (i4 > 0) {
                                }
                                if (measuredWidth < i4) {
                                }
                            }
                        }
                        z2 = false;
                        if (z2) {
                        }
                    }
                }
                i8 = (int) fraction3;
                if (i8 > 0) {
                }
            }
        }
        z = false;
        if (mode2 == Integer.MIN_VALUE) {
        }
        super.onMeasure(i, i2);
        measuredWidth = getMeasuredWidth();
        int makeMeasureSpec2 = View.MeasureSpec.makeMeasureSpec(measuredWidth, 1073741824);
        if (!z) {
            if (!z3) {
            }
            if (typedValue != null) {
                if (i3 != 5) {
                }
                i4 = (int) fraction;
                if (i4 > 0) {
                }
                if (measuredWidth < i4) {
                }
            }
        }
        z2 = false;
        if (z2) {
        }
    }

    public void setAttachListener(a aVar) {
        this.i = aVar;
    }
}