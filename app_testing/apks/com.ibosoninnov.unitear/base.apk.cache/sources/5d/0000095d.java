package c.d.c.h.e.f;

import com.google.firebase.crashlytics.internal.persistence.CrashlyticsReportPersistence;
import java.io.File;
import java.io.FilenameFilter;

/* compiled from: lambda */
/* loaded from: classes2.dex */
public final /* synthetic */ class d implements FilenameFilter {

    /* renamed from: a  reason: collision with root package name */
    public static final /* synthetic */ d f4433a = new d();

    @Override // java.io.FilenameFilter
    public final boolean accept(File file, String str) {
        int i = CrashlyticsReportPersistence.f5644a;
        return str.startsWith("event");
    }
}