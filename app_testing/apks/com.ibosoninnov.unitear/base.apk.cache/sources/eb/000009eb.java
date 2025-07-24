package c.e.b.ef;

import android.view.View;

/* compiled from: CategoryAdapter.java */
/* loaded from: classes2.dex */
public class b implements View.OnClickListener {

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ int f4710b;

    /* renamed from: c  reason: collision with root package name */
    public final /* synthetic */ c f4711c;

    public b(c cVar, int i) {
        this.f4711c = cVar;
        this.f4710b = i;
    }

    @Override // android.view.View.OnClickListener
    public void onClick(View view) {
        c cVar = this.f4711c;
        cVar.f4714c.h(cVar.f4712a.get(this.f4710b).name);
    }
}