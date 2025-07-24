package b.d.b.d1.k1.c;

import com.google.common.util.concurrent.ListenableFuture;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.concurrent.CancellationException;
import java.util.concurrent.ExecutionException;

/* compiled from: ListFuture.java */
/* loaded from: classes.dex */
public class k implements Runnable {

    /* renamed from: b  reason: collision with root package name */
    public final /* synthetic */ int f1564b;

    /* renamed from: c  reason: collision with root package name */
    public final /* synthetic */ ListenableFuture f1565c;

    /* renamed from: d  reason: collision with root package name */
    public final /* synthetic */ i f1566d;

    public k(i iVar, int i, ListenableFuture listenableFuture) {
        this.f1566d = iVar;
        this.f1564b = i;
        this.f1565c = listenableFuture;
    }

    @Override // java.lang.Runnable
    public void run() {
        b.g.a.b bVar;
        ArrayList arrayList;
        int decrementAndGet;
        i iVar = this.f1566d;
        int i = this.f1564b;
        ListenableFuture listenableFuture = this.f1565c;
        List<V> list = iVar.f1557c;
        if (!iVar.isDone() && list != 0) {
            try {
                try {
                    try {
                        try {
                            b.j.b.d.k(listenableFuture.isDone(), "Tried to set value from future which is not done");
                            list.set(i, g.b(listenableFuture));
                            decrementAndGet = iVar.f1559e.decrementAndGet();
                            b.j.b.d.k(decrementAndGet >= 0, "Less than 0 remaining futures");
                        } catch (RuntimeException e2) {
                            if (iVar.f1558d) {
                                iVar.f1561g.c(e2);
                            }
                            int decrementAndGet2 = iVar.f1559e.decrementAndGet();
                            b.j.b.d.k(decrementAndGet2 >= 0, "Less than 0 remaining futures");
                            if (decrementAndGet2 != 0) {
                                return;
                            }
                            Collection collection = iVar.f1557c;
                            if (collection != null) {
                                bVar = iVar.f1561g;
                                arrayList = new ArrayList(collection);
                            }
                        } catch (ExecutionException e3) {
                            if (iVar.f1558d) {
                                iVar.f1561g.c(e3.getCause());
                            }
                            int decrementAndGet3 = iVar.f1559e.decrementAndGet();
                            b.j.b.d.k(decrementAndGet3 >= 0, "Less than 0 remaining futures");
                            if (decrementAndGet3 != 0) {
                                return;
                            }
                            Collection collection2 = iVar.f1557c;
                            if (collection2 != null) {
                                bVar = iVar.f1561g;
                                arrayList = new ArrayList(collection2);
                            }
                        }
                    } catch (CancellationException unused) {
                        if (iVar.f1558d) {
                            iVar.cancel(false);
                        }
                        int decrementAndGet4 = iVar.f1559e.decrementAndGet();
                        b.j.b.d.k(decrementAndGet4 >= 0, "Less than 0 remaining futures");
                        if (decrementAndGet4 != 0) {
                            return;
                        }
                        Collection collection3 = iVar.f1557c;
                        if (collection3 != null) {
                            bVar = iVar.f1561g;
                            arrayList = new ArrayList(collection3);
                        }
                    }
                } catch (Error e4) {
                    iVar.f1561g.c(e4);
                    int decrementAndGet5 = iVar.f1559e.decrementAndGet();
                    b.j.b.d.k(decrementAndGet5 >= 0, "Less than 0 remaining futures");
                    if (decrementAndGet5 != 0) {
                        return;
                    }
                    Collection collection4 = iVar.f1557c;
                    if (collection4 != null) {
                        bVar = iVar.f1561g;
                        arrayList = new ArrayList(collection4);
                    }
                }
                if (decrementAndGet == 0) {
                    Collection collection5 = iVar.f1557c;
                    if (collection5 != null) {
                        bVar = iVar.f1561g;
                        arrayList = new ArrayList(collection5);
                        bVar.a(arrayList);
                        return;
                    }
                    b.j.b.d.k(iVar.isDone(), null);
                    return;
                }
                return;
            } catch (Throwable th) {
                int decrementAndGet6 = iVar.f1559e.decrementAndGet();
                b.j.b.d.k(decrementAndGet6 >= 0, "Less than 0 remaining futures");
                if (decrementAndGet6 == 0) {
                    Collection collection6 = iVar.f1557c;
                    if (collection6 != null) {
                        iVar.f1561g.a(new ArrayList(collection6));
                    } else {
                        b.j.b.d.k(iVar.isDone(), null);
                    }
                }
                throw th;
            }
        }
        b.j.b.d.k(iVar.f1558d, "Future was done before all dependencies completed");
    }
}