package b.b.h;

import android.content.Context;
import android.content.res.ColorStateList;
import android.content.res.TypedArray;
import android.graphics.Typeface;
import android.graphics.drawable.Drawable;
import android.util.AttributeSet;
import android.util.TypedValue;

/* compiled from: TintTypedArray.java */
/* loaded from: classes.dex */
public class y0 {

    /* renamed from: a  reason: collision with root package name */
    public final Context f971a;

    /* renamed from: b  reason: collision with root package name */
    public final TypedArray f972b;

    /* renamed from: c  reason: collision with root package name */
    public TypedValue f973c;

    public y0(Context context, TypedArray typedArray) {
        this.f971a = context;
        this.f972b = typedArray;
    }

    public static y0 q(Context context, AttributeSet attributeSet, int[] iArr) {
        return new y0(context, context.obtainStyledAttributes(attributeSet, iArr));
    }

    public static y0 r(Context context, AttributeSet attributeSet, int[] iArr, int i, int i2) {
        return new y0(context, context.obtainStyledAttributes(attributeSet, iArr, i, i2));
    }

    public boolean a(int i, boolean z) {
        return this.f972b.getBoolean(i, z);
    }

    public int b(int i, int i2) {
        return this.f972b.getColor(i, i2);
    }

    public ColorStateList c(int i) {
        int resourceId;
        if (this.f972b.hasValue(i) && (resourceId = this.f972b.getResourceId(i, 0)) != 0) {
            Context context = this.f971a;
            ThreadLocal<TypedValue> threadLocal = b.b.d.a.a.f630a;
            ColorStateList colorStateList = context.getColorStateList(resourceId);
            if (colorStateList != null) {
                return colorStateList;
            }
        }
        return this.f972b.getColorStateList(i);
    }

    public float d(int i, float f2) {
        return this.f972b.getDimension(i, f2);
    }

    public int e(int i, int i2) {
        return this.f972b.getDimensionPixelOffset(i, i2);
    }

    public int f(int i, int i2) {
        return this.f972b.getDimensionPixelSize(i, i2);
    }

    public Drawable g(int i) {
        int resourceId;
        if (this.f972b.hasValue(i) && (resourceId = this.f972b.getResourceId(i, 0)) != 0) {
            return b.b.d.a.a.a(this.f971a, resourceId);
        }
        return this.f972b.getDrawable(i);
    }

    public Drawable h(int i) {
        int resourceId;
        Drawable f2;
        if (!this.f972b.hasValue(i) || (resourceId = this.f972b.getResourceId(i, 0)) == 0) {
            return null;
        }
        j a2 = j.a();
        Context context = this.f971a;
        synchronized (a2) {
            f2 = a2.f864c.f(context, resourceId, true);
        }
        return f2;
    }

    public Typeface i(int i, int i2, b.j.c.b.e eVar) {
        int resourceId = this.f972b.getResourceId(i, 0);
        if (resourceId == 0) {
            return null;
        }
        if (this.f973c == null) {
            this.f973c = new TypedValue();
        }
        Context context = this.f971a;
        TypedValue typedValue = this.f973c;
        if (context.isRestricted()) {
            return null;
        }
        return b.j.c.b.f.f(context, resourceId, typedValue, i2, eVar, null, true, false);
    }

    public int j(int i, int i2) {
        return this.f972b.getInt(i, i2);
    }

    public int k(int i, int i2) {
        return this.f972b.getInteger(i, i2);
    }

    public int l(int i, int i2) {
        return this.f972b.getLayoutDimension(i, i2);
    }

    public int m(int i, int i2) {
        return this.f972b.getResourceId(i, i2);
    }

    public String n(int i) {
        return this.f972b.getString(i);
    }

    public CharSequence o(int i) {
        return this.f972b.getText(i);
    }

    public boolean p(int i) {
        return this.f972b.hasValue(i);
    }
}