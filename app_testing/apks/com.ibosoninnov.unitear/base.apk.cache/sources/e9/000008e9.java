package c.d.a.a.b.b.d;

import android.database.Cursor;
import com.google.android.datatransport.runtime.scheduling.persistence.SQLiteEventStore;

/* compiled from: lambda */
/* loaded from: classes.dex */
public final /* synthetic */ class z implements SQLiteEventStore.Function {

    /* renamed from: a  reason: collision with root package name */
    public static final /* synthetic */ z f4295a = new z();

    @Override // com.google.android.datatransport.runtime.scheduling.persistence.SQLiteEventStore.Function
    public final Object apply(Object obj) {
        int i = SQLiteEventStore.MAX_RETRIES;
        return Boolean.valueOf(((Cursor) obj).getCount() > 0);
    }
}