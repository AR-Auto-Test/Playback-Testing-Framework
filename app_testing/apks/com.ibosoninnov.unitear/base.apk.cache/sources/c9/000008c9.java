package c.d.a.a.b.b.d;

import com.google.android.datatransport.runtime.scheduling.persistence.SQLiteEventStore;
import com.google.android.datatransport.runtime.synchronization.SynchronizationException;

/* compiled from: lambda */
/* loaded from: classes.dex */
public final /* synthetic */ class a implements SQLiteEventStore.Function {

    /* renamed from: a  reason: collision with root package name */
    public static final /* synthetic */ a f4246a = new a();

    @Override // com.google.android.datatransport.runtime.scheduling.persistence.SQLiteEventStore.Function
    public final Object apply(Object obj) {
        int i = SQLiteEventStore.MAX_RETRIES;
        throw new SynchronizationException("Timed out while trying to open db.", (Throwable) obj);
    }
}