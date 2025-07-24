package b.d.b.d1;

import android.view.Surface;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ScheduledFuture;

/* compiled from: DeferrableSurfaces.java */
/* loaded from: classes.dex */
public class k0 implements b.d.b.d1.k1.c.d<List<Surface>> {

    /* renamed from: a  reason: collision with root package name */
    public final /* synthetic */ boolean f1512a;

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ b.g.a.b f1513b;

    /* renamed from: c  reason: collision with root package name */
    public final /* synthetic */ ScheduledFuture f1514c;

    public k0(boolean z, b.g.a.b bVar, ScheduledFuture scheduledFuture) {
        this.f1512a = z;
        this.f1513b = bVar;
        this.f1514c = scheduledFuture;
    }

    @Override // b.d.b.d1.k1.c.d
    public void onFailure(Throwable th) {
        this.f1513b.a(Collections.unmodifiableList(Collections.emptyList()));
        this.f1514c.cancel(true);
    }

    /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object] */
    @Override // b.d.b.d1.k1.c.d
    public void onSuccess(List<Surface> list) {
        ArrayList arrayList = new ArrayList(list);
        if (this.f1512a) {
            arrayList.removeAll(Collections.singleton(null));
        }
        this.f1513b.a(arrayList);
        this.f1514c.cancel(true);
    }
}