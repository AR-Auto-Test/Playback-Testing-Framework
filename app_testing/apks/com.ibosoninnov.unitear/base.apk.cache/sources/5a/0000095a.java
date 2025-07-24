package c.d.c.h.e.f;

import com.google.firebase.crashlytics.internal.persistence.CrashlyticsReportPersistence;
import java.io.File;
import java.util.Comparator;

/* compiled from: lambda */
/* loaded from: classes2.dex */
public final /* synthetic */ class a implements Comparator {

    /* renamed from: b  reason: collision with root package name */
    public static final /* synthetic */ a f4430b = new a();

    @Override // java.util.Comparator
    public final int compare(Object obj, Object obj2) {
        int i = CrashlyticsReportPersistence.f5644a;
        return ((File) obj2).getName().compareTo(((File) obj).getName());
    }
}