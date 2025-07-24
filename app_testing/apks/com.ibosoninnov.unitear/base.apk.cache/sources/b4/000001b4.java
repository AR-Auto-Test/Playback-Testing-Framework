package b.b.h;

import android.content.Context;
import android.content.res.ColorStateList;
import android.graphics.PorterDuff;
import android.graphics.drawable.Drawable;
import android.util.AttributeSet;
import android.view.View;
import java.util.concurrent.atomic.AtomicInteger;

/* compiled from: AppCompatBackgroundHelper.java */
/* loaded from: classes.dex */
public class e {

    /* renamed from: a  reason: collision with root package name */
    public final View f823a;

    /* renamed from: d  reason: collision with root package name */
    public w0 f826d;

    /* renamed from: e  reason: collision with root package name */
    public w0 f827e;

    /* renamed from: f  reason: collision with root package name */
    public w0 f828f;

    /* renamed from: c  reason: collision with root package name */
    public int f825c = -1;

    /* renamed from: b  reason: collision with root package name */
    public final j f824b = j.a();

    public e(View view) {
        this.f823a = view;
    }

    public void a() {
        Drawable background = this.f823a.getBackground();
        if (background != null) {
            boolean z = true;
            if (this.f826d != null) {
                if (this.f828f == null) {
                    this.f828f = new w0();
                }
                w0 w0Var = this.f828f;
                w0Var.f953a = null;
                w0Var.f956d = false;
                w0Var.f954b = null;
                w0Var.f955c = false;
                View view = this.f823a;
                AtomicInteger atomicInteger = b.j.j.q.f2214a;
                ColorStateList backgroundTintList = view.getBackgroundTintList();
                if (backgroundTintList != null) {
                    w0Var.f956d = true;
                    w0Var.f953a = backgroundTintList;
                }
                PorterDuff.Mode backgroundTintMode = this.f823a.getBackgroundTintMode();
                if (backgroundTintMode != null) {
                    w0Var.f955c = true;
                    w0Var.f954b = backgroundTintMode;
                }
                if (w0Var.f956d || w0Var.f955c) {
                    j.f(background, w0Var, this.f823a.getDrawableState());
                } else {
                    z = false;
                }
                if (z) {
                    return;
                }
            }
            w0 w0Var2 = this.f827e;
            if (w0Var2 != null) {
                j.f(background, w0Var2, this.f823a.getDrawableState());
                return;
            }
            w0 w0Var3 = this.f826d;
            if (w0Var3 != null) {
                j.f(background, w0Var3, this.f823a.getDrawableState());
            }
        }
    }

    public ColorStateList b() {
        w0 w0Var = this.f827e;
        if (w0Var != null) {
            return w0Var.f953a;
        }
        return null;
    }

    public PorterDuff.Mode c() {
        w0 w0Var = this.f827e;
        if (w0Var != null) {
            return w0Var.f954b;
        }
        return null;
    }

    public void d(AttributeSet attributeSet, int i) {
        Context context = this.f823a.getContext();
        int[] iArr = b.b.b.A;
        y0 r = y0.r(context, attributeSet, iArr, i, 0);
        View view = this.f823a;
        b.j.j.q.m(view, view.getContext(), iArr, attributeSet, r.f972b, i, 0);
        try {
            if (r.p(0)) {
                this.f825c = r.m(0, -1);
                ColorStateList d2 = this.f824b.d(this.f823a.getContext(), this.f825c);
                if (d2 != null) {
                    g(d2);
                }
            }
            if (r.p(1)) {
                this.f823a.setBackgroundTintList(r.c(1));
            }
            if (r.p(2)) {
                this.f823a.setBackgroundTintMode(e0.c(r.j(2, -1), null));
            }
            r.f972b.recycle();
        } catch (Throwable th) {
            r.f972b.recycle();
            throw th;
        }
    }

    public void e() {
        this.f825c = -1;
        g(null);
        a();
    }

    public void f(int i) {
        this.f825c = i;
        j jVar = this.f824b;
        g(jVar != null ? jVar.d(this.f823a.getContext(), i) : null);
        a();
    }

    public void g(ColorStateList colorStateList) {
        if (colorStateList != null) {
            if (this.f826d == null) {
                this.f826d = new w0();
            }
            w0 w0Var = this.f826d;
            w0Var.f953a = colorStateList;
            w0Var.f956d = true;
        } else {
            this.f826d = null;
        }
        a();
    }

    public void h(ColorStateList colorStateList) {
        if (this.f827e == null) {
            this.f827e = new w0();
        }
        w0 w0Var = this.f827e;
        w0Var.f953a = colorStateList;
        w0Var.f956d = true;
        a();
    }

    public void i(PorterDuff.Mode mode) {
        if (this.f827e == null) {
            this.f827e = new w0();
        }
        w0 w0Var = this.f827e;
        w0Var.f954b = mode;
        w0Var.f955c = true;
        a();
    }
}