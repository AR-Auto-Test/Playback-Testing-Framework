package b.b.g.i;

import android.content.Context;
import android.graphics.Point;
import android.graphics.Rect;
import android.view.Display;
import android.view.Gravity;
import android.view.View;
import android.view.WindowManager;
import android.widget.PopupWindow;
import b.b.g.i.m;
import com.ibosoninnov.unitear.R;
import java.util.concurrent.atomic.AtomicInteger;

/* compiled from: MenuPopupHelper.java */
/* loaded from: classes.dex */
public class l {

    /* renamed from: a  reason: collision with root package name */
    public final Context f750a;

    /* renamed from: b  reason: collision with root package name */
    public final g f751b;

    /* renamed from: c  reason: collision with root package name */
    public final boolean f752c;

    /* renamed from: d  reason: collision with root package name */
    public final int f753d;

    /* renamed from: e  reason: collision with root package name */
    public final int f754e;

    /* renamed from: f  reason: collision with root package name */
    public View f755f;

    /* renamed from: h  reason: collision with root package name */
    public boolean f757h;
    public m.a i;
    public k j;
    public PopupWindow.OnDismissListener k;

    /* renamed from: g  reason: collision with root package name */
    public int f756g = 8388611;
    public final PopupWindow.OnDismissListener l = new a();

    /* compiled from: MenuPopupHelper.java */
    /* loaded from: classes.dex */
    public class a implements PopupWindow.OnDismissListener {
        public a() {
        }

        @Override // android.widget.PopupWindow.OnDismissListener
        public void onDismiss() {
            l.this.c();
        }
    }

    public l(Context context, g gVar, View view, boolean z, int i, int i2) {
        this.f750a = context;
        this.f751b = gVar;
        this.f755f = view;
        this.f752c = z;
        this.f753d = i;
        this.f754e = i2;
    }

    public k a() {
        k qVar;
        if (this.j == null) {
            Display defaultDisplay = ((WindowManager) this.f750a.getSystemService("window")).getDefaultDisplay();
            Point point = new Point();
            defaultDisplay.getRealSize(point);
            if (Math.min(point.x, point.y) >= this.f750a.getResources().getDimensionPixelSize(R.dimen.abc_cascading_menus_min_smallest_width)) {
                qVar = new d(this.f750a, this.f755f, this.f753d, this.f754e, this.f752c);
            } else {
                qVar = new q(this.f750a, this.f751b, this.f755f, this.f753d, this.f754e, this.f752c);
            }
            qVar.b(this.f751b);
            qVar.i(this.l);
            qVar.d(this.f755f);
            qVar.setCallback(this.i);
            qVar.e(this.f757h);
            qVar.f(this.f756g);
            this.j = qVar;
        }
        return this.j;
    }

    public boolean b() {
        k kVar = this.j;
        return kVar != null && kVar.a();
    }

    public void c() {
        this.j = null;
        PopupWindow.OnDismissListener onDismissListener = this.k;
        if (onDismissListener != null) {
            onDismissListener.onDismiss();
        }
    }

    public void d(m.a aVar) {
        this.i = aVar;
        k kVar = this.j;
        if (kVar != null) {
            kVar.setCallback(aVar);
        }
    }

    public final void e(int i, int i2, boolean z, boolean z2) {
        k a2 = a();
        a2.j(z2);
        if (z) {
            int i3 = this.f756g;
            View view = this.f755f;
            AtomicInteger atomicInteger = b.j.j.q.f2214a;
            if ((Gravity.getAbsoluteGravity(i3, view.getLayoutDirection()) & 7) == 5) {
                i -= this.f755f.getWidth();
            }
            a2.g(i);
            a2.k(i2);
            int i4 = (int) ((this.f750a.getResources().getDisplayMetrics().density * 48.0f) / 2.0f);
            a2.f749b = new Rect(i - i4, i2 - i4, i + i4, i2 + i4);
        }
        a2.show();
    }

    public boolean f() {
        if (b()) {
            return true;
        }
        if (this.f755f == null) {
            return false;
        }
        e(0, 0, false, false);
        return true;
    }
}