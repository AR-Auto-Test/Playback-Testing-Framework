package b.b.c;

import android.content.Context;
import android.content.res.TypedArray;
import android.util.AttributeSet;
import android.view.ViewGroup;

/* compiled from: ActionBar.java */
/* loaded from: classes.dex */
public abstract class a {

    /* compiled from: ActionBar.java */
    /* loaded from: classes.dex */
    public interface b {
        void a(boolean z);
    }

    /* compiled from: ActionBar.java */
    @Deprecated
    /* loaded from: classes.dex */
    public static abstract class c {
        public abstract void a();
    }

    public abstract void a(boolean z);

    public abstract Context b();

    public abstract void c(boolean z);

    /* compiled from: ActionBar.java */
    /* renamed from: b.b.c.a$a  reason: collision with other inner class name */
    /* loaded from: classes.dex */
    public static class C0006a extends ViewGroup.MarginLayoutParams {

        /* renamed from: a  reason: collision with root package name */
        public int f549a;

        public C0006a(Context context, AttributeSet attributeSet) {
            super(context, attributeSet);
            this.f549a = 0;
            TypedArray obtainStyledAttributes = context.obtainStyledAttributes(attributeSet, b.b.b.f542b);
            this.f549a = obtainStyledAttributes.getInt(0, 0);
            obtainStyledAttributes.recycle();
        }

        public C0006a(int i, int i2) {
            super(i, i2);
            this.f549a = 0;
            this.f549a = 8388627;
        }

        public C0006a(C0006a c0006a) {
            super((ViewGroup.MarginLayoutParams) c0006a);
            this.f549a = 0;
            this.f549a = c0006a.f549a;
        }

        public C0006a(ViewGroup.LayoutParams layoutParams) {
            super(layoutParams);
            this.f549a = 0;
        }
    }
}