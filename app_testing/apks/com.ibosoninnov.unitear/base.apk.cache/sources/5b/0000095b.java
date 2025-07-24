package c.d.c.h.e.f;

import com.google.firebase.crashlytics.internal.persistence.CrashlyticsReportPersistence;
import java.io.File;
import java.io.FilenameFilter;

/* compiled from: lambda */
/* loaded from: classes2.dex */
public final /* synthetic */ class b implements FilenameFilter {

    /* renamed from: a  reason: collision with root package name */
    public static final /* synthetic */ b f4431a = new b();

    @Override // java.io.FilenameFilter
    public final boolean accept(File file, String str) {
        boolean isNormalPriorityEventFile;
        isNormalPriorityEventFile = CrashlyticsReportPersistence.isNormalPriorityEventFile(file, str);
        return isNormalPriorityEventFile;
    }
}