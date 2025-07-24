package b.b.h;

import android.view.View;
import android.view.Window;

/* compiled from: ToolbarWidgetWrapper.java */
/* loaded from: classes.dex */
public class z0 implements View.OnClickListener {

    /* renamed from: b  reason: collision with root package name */
    public final b.b.g.i.a f974b;

    /* renamed from: c  reason: collision with root package name */
    public final /* synthetic */ a1 f975c;

    public z0(a1 a1Var) {
        this.f975c = a1Var;
        this.f974b = new b.b.g.i.a(a1Var.f787a.getContext(), 0, 16908332, 0, a1Var.i);
    }

    @Override // android.view.View.OnClickListener
    public void onClick(View view) {
        a1 a1Var = this.f975c;
        Window.Callback callback = a1Var.l;
        if (callback == null || !a1Var.m) {
            return;
        }
        callback.onMenuItemSelected(0, this.f974b);
    }
}