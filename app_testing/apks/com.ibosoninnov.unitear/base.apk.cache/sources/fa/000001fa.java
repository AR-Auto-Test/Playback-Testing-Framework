package b.b.h;

import android.annotation.SuppressLint;
import android.content.Context;
import android.content.res.ColorStateList;
import android.content.res.Resources;
import android.content.res.TypedArray;
import android.graphics.PorterDuff;
import android.graphics.Typeface;
import android.graphics.drawable.Drawable;
import android.os.Build;
import android.os.LocaleList;
import android.text.method.PasswordTransformationMethod;
import android.util.AttributeSet;
import android.util.DisplayMetrics;
import android.util.TypedValue;
import android.widget.TextView;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import java.lang.ref.WeakReference;
import java.util.Arrays;
import java.util.Objects;

/* compiled from: AppCompatTextHelper.java */
/* loaded from: classes.dex */
public class y {

    /* renamed from: a  reason: collision with root package name */
    public final TextView f959a;

    /* renamed from: b  reason: collision with root package name */
    public w0 f960b;

    /* renamed from: c  reason: collision with root package name */
    public w0 f961c;

    /* renamed from: d  reason: collision with root package name */
    public w0 f962d;

    /* renamed from: e  reason: collision with root package name */
    public w0 f963e;

    /* renamed from: f  reason: collision with root package name */
    public w0 f964f;

    /* renamed from: g  reason: collision with root package name */
    public w0 f965g;

    /* renamed from: h  reason: collision with root package name */
    public w0 f966h;
    public final a0 i;
    public int j = 0;
    public int k = -1;
    public Typeface l;
    public boolean m;

    /* compiled from: AppCompatTextHelper.java */
    /* loaded from: classes.dex */
    public class a extends b.j.c.b.e {

        /* renamed from: a  reason: collision with root package name */
        public final /* synthetic */ int f967a;

        /* renamed from: b  reason: collision with root package name */
        public final /* synthetic */ int f968b;

        /* renamed from: c  reason: collision with root package name */
        public final /* synthetic */ WeakReference f969c;

        public a(int i, int i2, WeakReference weakReference) {
            this.f967a = i;
            this.f968b = i2;
            this.f969c = weakReference;
        }

        @Override // b.j.c.b.e
        public void onFontRetrievalFailed(int i) {
        }

        @Override // b.j.c.b.e
        public void onFontRetrieved(Typeface typeface) {
            int i;
            if (Build.VERSION.SDK_INT >= 28 && (i = this.f967a) != -1) {
                typeface = Typeface.create(typeface, i, (this.f968b & 2) != 0);
            }
            y yVar = y.this;
            WeakReference weakReference = this.f969c;
            if (yVar.m) {
                yVar.l = typeface;
                TextView textView = (TextView) weakReference.get();
                if (textView != null) {
                    textView.setTypeface(typeface, yVar.j);
                }
            }
        }
    }

    public y(TextView textView) {
        this.f959a = textView;
        this.i = new a0(textView);
    }

    public static w0 c(Context context, j jVar, int i) {
        ColorStateList d2 = jVar.d(context, i);
        if (d2 != null) {
            w0 w0Var = new w0();
            w0Var.f956d = true;
            w0Var.f953a = d2;
            return w0Var;
        }
        return null;
    }

    public final void a(Drawable drawable, w0 w0Var) {
        if (drawable == null || w0Var == null) {
            return;
        }
        j.f(drawable, w0Var, this.f959a.getDrawableState());
    }

    public void b() {
        if (this.f960b != null || this.f961c != null || this.f962d != null || this.f963e != null) {
            Drawable[] compoundDrawables = this.f959a.getCompoundDrawables();
            a(compoundDrawables[0], this.f960b);
            a(compoundDrawables[1], this.f961c);
            a(compoundDrawables[2], this.f962d);
            a(compoundDrawables[3], this.f963e);
        }
        if (this.f964f == null && this.f965g == null) {
            return;
        }
        Drawable[] compoundDrawablesRelative = this.f959a.getCompoundDrawablesRelative();
        a(compoundDrawablesRelative[0], this.f964f);
        a(compoundDrawablesRelative[2], this.f965g);
    }

    public boolean d() {
        a0 a0Var = this.i;
        return a0Var.i() && a0Var.f782d != 0;
    }

    /* JADX WARN: Code restructure failed: missing block: B:208:0x036e, code lost:
        if (r3 != null) goto L151;
     */
    @SuppressLint({"NewApi"})
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public void e(AttributeSet attributeSet, int i) {
        boolean z;
        boolean z2;
        String str;
        String str2;
        int i2;
        Drawable drawable;
        int i3;
        ColorStateList colorStateList;
        int resourceId;
        int i4;
        int resourceId2;
        Context context = this.f959a.getContext();
        j a2 = j.a();
        int[] iArr = b.b.b.f548h;
        y0 r = y0.r(context, attributeSet, iArr, i, 0);
        TextView textView = this.f959a;
        b.j.j.q.m(textView, textView.getContext(), iArr, attributeSet, r.f972b, i, 0);
        int m = r.m(0, -1);
        if (r.p(3)) {
            this.f960b = c(context, a2, r.m(3, 0));
        }
        if (r.p(1)) {
            this.f961c = c(context, a2, r.m(1, 0));
        }
        if (r.p(4)) {
            this.f962d = c(context, a2, r.m(4, 0));
        }
        if (r.p(2)) {
            this.f963e = c(context, a2, r.m(2, 0));
        }
        int i5 = Build.VERSION.SDK_INT;
        if (r.p(5)) {
            this.f964f = c(context, a2, r.m(5, 0));
        }
        if (r.p(6)) {
            this.f965g = c(context, a2, r.m(6, 0));
        }
        r.f972b.recycle();
        boolean z3 = this.f959a.getTransformationMethod() instanceof PasswordTransformationMethod;
        if (m != -1) {
            TypedArray obtainStyledAttributes = context.obtainStyledAttributes(m, b.b.b.x);
            y0 y0Var = new y0(context, obtainStyledAttributes);
            if (z3 || !y0Var.p(14)) {
                z = false;
                z2 = false;
            } else {
                z = y0Var.a(14, false);
                z2 = true;
            }
            l(context, y0Var);
            str2 = y0Var.p(15) ? y0Var.n(15) : null;
            str = (i5 < 26 || !y0Var.p(13)) ? null : y0Var.n(13);
            obtainStyledAttributes.recycle();
        } else {
            z = false;
            z2 = false;
            str = null;
            str2 = null;
        }
        TypedArray obtainStyledAttributes2 = context.obtainStyledAttributes(attributeSet, b.b.b.x, i, 0);
        y0 y0Var2 = new y0(context, obtainStyledAttributes2);
        if (!z3 && y0Var2.p(14)) {
            z = y0Var2.a(14, false);
            z2 = true;
        }
        if (y0Var2.p(15)) {
            str2 = y0Var2.n(15);
        }
        if (i5 >= 26 && y0Var2.p(13)) {
            str = y0Var2.n(13);
        }
        if (i5 >= 28 && y0Var2.p(0) && y0Var2.f(0, -1) == 0) {
            this.f959a.setTextSize(0, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
        }
        l(context, y0Var2);
        obtainStyledAttributes2.recycle();
        if (!z3 && z2) {
            this.f959a.setAllCaps(z);
        }
        Typeface typeface = this.l;
        if (typeface != null) {
            if (this.k == -1) {
                this.f959a.setTypeface(typeface, this.j);
            } else {
                this.f959a.setTypeface(typeface);
            }
        }
        if (str != null) {
            this.f959a.setFontVariationSettings(str);
        }
        if (str2 != null) {
            this.f959a.setTextLocales(LocaleList.forLanguageTags(str2));
        }
        a0 a0Var = this.i;
        Context context2 = a0Var.m;
        int[] iArr2 = b.b.b.i;
        TypedArray obtainStyledAttributes3 = context2.obtainStyledAttributes(attributeSet, iArr2, i, 0);
        TextView textView2 = a0Var.l;
        b.j.j.q.m(textView2, textView2.getContext(), iArr2, attributeSet, obtainStyledAttributes3, i, 0);
        if (obtainStyledAttributes3.hasValue(5)) {
            a0Var.f782d = obtainStyledAttributes3.getInt(5, 0);
        }
        float dimension = obtainStyledAttributes3.hasValue(4) ? obtainStyledAttributes3.getDimension(4, -1.0f) : -1.0f;
        float dimension2 = obtainStyledAttributes3.hasValue(2) ? obtainStyledAttributes3.getDimension(2, -1.0f) : -1.0f;
        float dimension3 = obtainStyledAttributes3.hasValue(1) ? obtainStyledAttributes3.getDimension(1, -1.0f) : -1.0f;
        if (obtainStyledAttributes3.hasValue(3) && (resourceId2 = obtainStyledAttributes3.getResourceId(3, 0)) > 0) {
            TypedArray obtainTypedArray = obtainStyledAttributes3.getResources().obtainTypedArray(resourceId2);
            int length = obtainTypedArray.length();
            int[] iArr3 = new int[length];
            if (length > 0) {
                for (int i6 = 0; i6 < length; i6++) {
                    iArr3[i6] = obtainTypedArray.getDimensionPixelSize(i6, -1);
                }
                a0Var.i = a0Var.b(iArr3);
                a0Var.h();
            }
            obtainTypedArray.recycle();
        }
        obtainStyledAttributes3.recycle();
        if (a0Var.i()) {
            if (a0Var.f782d == 1) {
                if (!a0Var.j) {
                    DisplayMetrics displayMetrics = a0Var.m.getResources().getDisplayMetrics();
                    if (dimension2 == -1.0f) {
                        i4 = 2;
                        dimension2 = TypedValue.applyDimension(2, 12.0f, displayMetrics);
                    } else {
                        i4 = 2;
                    }
                    if (dimension3 == -1.0f) {
                        dimension3 = TypedValue.applyDimension(i4, 112.0f, displayMetrics);
                    }
                    if (dimension == -1.0f) {
                        dimension = 1.0f;
                    }
                    a0Var.j(dimension2, dimension3, dimension);
                }
                a0Var.g();
            }
        } else {
            a0Var.f782d = 0;
        }
        if (b.j.k.b.f2297a) {
            a0 a0Var2 = this.i;
            if (a0Var2.f782d != 0) {
                int[] iArr4 = a0Var2.i;
                if (iArr4.length > 0) {
                    if (this.f959a.getAutoSizeStepGranularity() != -1.0f) {
                        this.f959a.setAutoSizeTextTypeUniformWithConfiguration(Math.round(this.i.f785g), Math.round(this.i.f786h), Math.round(this.i.f784f), 0);
                    } else {
                        this.f959a.setAutoSizeTextTypeUniformWithPresetSizes(iArr4, 0);
                    }
                }
            }
        }
        TypedArray obtainStyledAttributes4 = context.obtainStyledAttributes(attributeSet, b.b.b.i);
        int resourceId3 = obtainStyledAttributes4.getResourceId(8, -1);
        if (resourceId3 != -1) {
            drawable = a2.b(context, resourceId3);
            i2 = 13;
        } else {
            i2 = 13;
            drawable = null;
        }
        int resourceId4 = obtainStyledAttributes4.getResourceId(i2, -1);
        Drawable b2 = resourceId4 != -1 ? a2.b(context, resourceId4) : null;
        int resourceId5 = obtainStyledAttributes4.getResourceId(9, -1);
        Drawable b3 = resourceId5 != -1 ? a2.b(context, resourceId5) : null;
        int resourceId6 = obtainStyledAttributes4.getResourceId(6, -1);
        Drawable b4 = resourceId6 != -1 ? a2.b(context, resourceId6) : null;
        int resourceId7 = obtainStyledAttributes4.getResourceId(10, -1);
        Drawable b5 = resourceId7 != -1 ? a2.b(context, resourceId7) : null;
        int resourceId8 = obtainStyledAttributes4.getResourceId(7, -1);
        Drawable b6 = resourceId8 != -1 ? a2.b(context, resourceId8) : null;
        if (b5 != null || b6 != null) {
            Drawable[] compoundDrawablesRelative = this.f959a.getCompoundDrawablesRelative();
            TextView textView3 = this.f959a;
            if (b5 == null) {
                b5 = compoundDrawablesRelative[0];
            }
            if (b2 == null) {
                b2 = compoundDrawablesRelative[1];
            }
            if (b6 == null) {
                b6 = compoundDrawablesRelative[2];
            }
            if (b4 == null) {
                b4 = compoundDrawablesRelative[3];
            }
            textView3.setCompoundDrawablesRelativeWithIntrinsicBounds(b5, b2, b6, b4);
        } else if (drawable != null || b2 != null || b3 != null || b4 != null) {
            Drawable[] compoundDrawablesRelative2 = this.f959a.getCompoundDrawablesRelative();
            if (compoundDrawablesRelative2[0] == null && compoundDrawablesRelative2[2] == null) {
                Drawable[] compoundDrawables = this.f959a.getCompoundDrawables();
                TextView textView4 = this.f959a;
                if (drawable == null) {
                    drawable = compoundDrawables[0];
                }
                if (b2 == null) {
                    b2 = compoundDrawables[1];
                }
                if (b3 == null) {
                    b3 = compoundDrawables[2];
                }
                if (b4 == null) {
                    b4 = compoundDrawables[3];
                }
                textView4.setCompoundDrawablesWithIntrinsicBounds(drawable, b2, b3, b4);
            } else {
                TextView textView5 = this.f959a;
                Drawable drawable2 = compoundDrawablesRelative2[0];
                if (b2 == null) {
                    b2 = compoundDrawablesRelative2[1];
                }
                Drawable drawable3 = compoundDrawablesRelative2[2];
                if (b4 == null) {
                    b4 = compoundDrawablesRelative2[3];
                }
                textView5.setCompoundDrawablesRelativeWithIntrinsicBounds(drawable2, b2, drawable3, b4);
            }
        }
        if (obtainStyledAttributes4.hasValue(11)) {
            if (obtainStyledAttributes4.hasValue(11) && (resourceId = obtainStyledAttributes4.getResourceId(11, 0)) != 0) {
                ThreadLocal<TypedValue> threadLocal = b.b.d.a.a.f630a;
                colorStateList = context.getColorStateList(resourceId);
            }
            colorStateList = obtainStyledAttributes4.getColorStateList(11);
            TextView textView6 = this.f959a;
            Objects.requireNonNull(textView6);
            textView6.setCompoundDrawableTintList(colorStateList);
        }
        if (obtainStyledAttributes4.hasValue(12)) {
            i3 = -1;
            PorterDuff.Mode c2 = e0.c(obtainStyledAttributes4.getInt(12, -1), null);
            TextView textView7 = this.f959a;
            Objects.requireNonNull(textView7);
            textView7.setCompoundDrawableTintMode(c2);
        } else {
            i3 = -1;
        }
        int dimensionPixelSize = obtainStyledAttributes4.getDimensionPixelSize(14, i3);
        int dimensionPixelSize2 = obtainStyledAttributes4.getDimensionPixelSize(17, i3);
        int dimensionPixelSize3 = obtainStyledAttributes4.getDimensionPixelSize(18, i3);
        obtainStyledAttributes4.recycle();
        if (dimensionPixelSize != i3) {
            b.j.b.d.M(this.f959a, dimensionPixelSize);
        }
        if (dimensionPixelSize2 != i3) {
            b.j.b.d.N(this.f959a, dimensionPixelSize2);
        }
        if (dimensionPixelSize3 != i3) {
            b.j.b.d.O(this.f959a, dimensionPixelSize3);
        }
    }

    public void f(Context context, int i) {
        String n;
        TypedArray obtainStyledAttributes = context.obtainStyledAttributes(i, b.b.b.x);
        y0 y0Var = new y0(context, obtainStyledAttributes);
        if (y0Var.p(14)) {
            this.f959a.setAllCaps(y0Var.a(14, false));
        }
        int i2 = Build.VERSION.SDK_INT;
        if (y0Var.p(0) && y0Var.f(0, -1) == 0) {
            this.f959a.setTextSize(0, StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
        }
        l(context, y0Var);
        if (i2 >= 26 && y0Var.p(13) && (n = y0Var.n(13)) != null) {
            this.f959a.setFontVariationSettings(n);
        }
        obtainStyledAttributes.recycle();
        Typeface typeface = this.l;
        if (typeface != null) {
            this.f959a.setTypeface(typeface, this.j);
        }
    }

    public void g(int i, int i2, int i3, int i4) {
        a0 a0Var = this.i;
        if (a0Var.i()) {
            DisplayMetrics displayMetrics = a0Var.m.getResources().getDisplayMetrics();
            a0Var.j(TypedValue.applyDimension(i4, i, displayMetrics), TypedValue.applyDimension(i4, i2, displayMetrics), TypedValue.applyDimension(i4, i3, displayMetrics));
            if (a0Var.g()) {
                a0Var.a();
            }
        }
    }

    public void h(int[] iArr, int i) {
        a0 a0Var = this.i;
        if (a0Var.i()) {
            int length = iArr.length;
            if (length > 0) {
                int[] iArr2 = new int[length];
                if (i == 0) {
                    iArr2 = Arrays.copyOf(iArr, length);
                } else {
                    DisplayMetrics displayMetrics = a0Var.m.getResources().getDisplayMetrics();
                    for (int i2 = 0; i2 < length; i2++) {
                        iArr2[i2] = Math.round(TypedValue.applyDimension(i, iArr[i2], displayMetrics));
                    }
                }
                a0Var.i = a0Var.b(iArr2);
                if (!a0Var.h()) {
                    StringBuilder x = c.b.a.a.a.x("None of the preset sizes is valid: ");
                    x.append(Arrays.toString(iArr));
                    throw new IllegalArgumentException(x.toString());
                }
            } else {
                a0Var.j = false;
            }
            if (a0Var.g()) {
                a0Var.a();
            }
        }
    }

    public void i(int i) {
        a0 a0Var = this.i;
        if (a0Var.i()) {
            if (i == 0) {
                a0Var.f782d = 0;
                a0Var.f785g = -1.0f;
                a0Var.f786h = -1.0f;
                a0Var.f784f = -1.0f;
                a0Var.i = new int[0];
                a0Var.f783e = false;
            } else if (i == 1) {
                DisplayMetrics displayMetrics = a0Var.m.getResources().getDisplayMetrics();
                a0Var.j(TypedValue.applyDimension(2, 12.0f, displayMetrics), TypedValue.applyDimension(2, 112.0f, displayMetrics), 1.0f);
                if (a0Var.g()) {
                    a0Var.a();
                }
            } else {
                throw new IllegalArgumentException(c.b.a.a.a.j("Unknown auto-size text type: ", i));
            }
        }
    }

    public void j(ColorStateList colorStateList) {
        if (this.f966h == null) {
            this.f966h = new w0();
        }
        w0 w0Var = this.f966h;
        w0Var.f953a = colorStateList;
        w0Var.f956d = colorStateList != null;
        this.f960b = w0Var;
        this.f961c = w0Var;
        this.f962d = w0Var;
        this.f963e = w0Var;
        this.f964f = w0Var;
        this.f965g = w0Var;
    }

    public void k(PorterDuff.Mode mode) {
        if (this.f966h == null) {
            this.f966h = new w0();
        }
        w0 w0Var = this.f966h;
        w0Var.f954b = mode;
        w0Var.f955c = mode != null;
        this.f960b = w0Var;
        this.f961c = w0Var;
        this.f962d = w0Var;
        this.f963e = w0Var;
        this.f964f = w0Var;
        this.f965g = w0Var;
    }

    public final void l(Context context, y0 y0Var) {
        String n;
        this.j = y0Var.j(2, this.j);
        int i = Build.VERSION.SDK_INT;
        if (i >= 28) {
            int j = y0Var.j(11, -1);
            this.k = j;
            if (j != -1) {
                this.j = (this.j & 2) | 0;
            }
        }
        if (!y0Var.p(10) && !y0Var.p(12)) {
            if (y0Var.p(1)) {
                this.m = false;
                int j2 = y0Var.j(1, 1);
                if (j2 == 1) {
                    this.l = Typeface.SANS_SERIF;
                    return;
                } else if (j2 == 2) {
                    this.l = Typeface.SERIF;
                    return;
                } else if (j2 != 3) {
                    return;
                } else {
                    this.l = Typeface.MONOSPACE;
                    return;
                }
            }
            return;
        }
        this.l = null;
        int i2 = y0Var.p(12) ? 12 : 10;
        int i3 = this.k;
        int i4 = this.j;
        if (!context.isRestricted()) {
            try {
                Typeface i5 = y0Var.i(i2, this.j, new a(i3, i4, new WeakReference(this.f959a)));
                if (i5 != null) {
                    if (i >= 28 && this.k != -1) {
                        this.l = Typeface.create(Typeface.create(i5, 0), this.k, (this.j & 2) != 0);
                    } else {
                        this.l = i5;
                    }
                }
                this.m = this.l == null;
            } catch (Resources.NotFoundException | UnsupportedOperationException unused) {
            }
        }
        if (this.l != null || (n = y0Var.n(i2)) == null) {
            return;
        }
        if (Build.VERSION.SDK_INT >= 28 && this.k != -1) {
            this.l = Typeface.create(Typeface.create(n, 0), this.k, (this.j & 2) != 0);
        } else {
            this.l = Typeface.create(n, this.j);
        }
    }
}