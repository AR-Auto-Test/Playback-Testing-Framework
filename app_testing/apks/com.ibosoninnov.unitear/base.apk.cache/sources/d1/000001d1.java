package b.b.h;

import android.content.Context;
import android.content.res.ColorStateList;
import android.graphics.PorterDuff;
import android.graphics.drawable.Drawable;
import android.util.AttributeSet;
import android.widget.ImageView;

/* compiled from: AppCompatImageHelper.java */
/* loaded from: classes.dex */
public class m {

    /* renamed from: a  reason: collision with root package name */
    public final ImageView f884a;

    /* renamed from: b  reason: collision with root package name */
    public w0 f885b;

    public m(ImageView imageView) {
        this.f884a = imageView;
    }

    public void a() {
        w0 w0Var;
        Drawable drawable = this.f884a.getDrawable();
        if (drawable != null) {
            int[] iArr = e0.f829a;
        }
        if (drawable == null || (w0Var = this.f885b) == null) {
            return;
        }
        j.f(drawable, w0Var, this.f884a.getDrawableState());
    }

    public void b(AttributeSet attributeSet, int i) {
        int m;
        Context context = this.f884a.getContext();
        int[] iArr = b.b.b.f546f;
        y0 r = y0.r(context, attributeSet, iArr, i, 0);
        ImageView imageView = this.f884a;
        b.j.j.q.m(imageView, imageView.getContext(), iArr, attributeSet, r.f972b, i, 0);
        try {
            Drawable drawable = this.f884a.getDrawable();
            if (drawable == null && (m = r.m(1, -1)) != -1 && (drawable = b.b.d.a.a.a(this.f884a.getContext(), m)) != null) {
                this.f884a.setImageDrawable(drawable);
            }
            if (drawable != null) {
                int[] iArr2 = e0.f829a;
            }
            if (r.p(2)) {
                this.f884a.setImageTintList(r.c(2));
            }
            if (r.p(3)) {
                this.f884a.setImageTintMode(e0.c(r.j(3, -1), null));
            }
            r.f972b.recycle();
        } catch (Throwable th) {
            r.f972b.recycle();
            throw th;
        }
    }

    public void c(int i) {
        if (i != 0) {
            Drawable a2 = b.b.d.a.a.a(this.f884a.getContext(), i);
            if (a2 != null) {
                int[] iArr = e0.f829a;
            }
            this.f884a.setImageDrawable(a2);
        } else {
            this.f884a.setImageDrawable(null);
        }
        a();
    }

    public void d(ColorStateList colorStateList) {
        if (this.f885b == null) {
            this.f885b = new w0();
        }
        w0 w0Var = this.f885b;
        w0Var.f953a = colorStateList;
        w0Var.f956d = true;
        a();
    }

    public void e(PorterDuff.Mode mode) {
        if (this.f885b == null) {
            this.f885b = new w0();
        }
        w0 w0Var = this.f885b;
        w0Var.f954b = mode;
        w0Var.f955c = true;
        a();
    }
}