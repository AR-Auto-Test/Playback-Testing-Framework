package c.d.c.h.e.f;

import com.google.firebase.crashlytics.internal.persistence.CrashlyticsReportPersistence;
import java.io.File;
import java.util.Comparator;

/* compiled from: lambda */
/* loaded from: classes2.dex */
public final /* synthetic */ class c implements Comparator {

    /* renamed from: b  reason: collision with root package name */
    public static final /* synthetic */ c f4432b = new c();

    @Override // java.util.Comparator
    public final int compare(Object obj, Object obj2) {
        int oldestEventFileFirst;
        oldestEventFileFirst = CrashlyticsReportPersistence.oldestEventFileFirst((File) obj, (File) obj2);
        return oldestEventFileFirst;
    }
}