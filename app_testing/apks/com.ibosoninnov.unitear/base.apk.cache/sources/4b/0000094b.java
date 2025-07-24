package c.d.c.h.e.c;

import com.google.firebase.crashlytics.internal.common.CrashlyticsController;
import java.io.File;
import java.io.FilenameFilter;

/* compiled from: lambda */
/* loaded from: classes2.dex */
public final /* synthetic */ class a implements FilenameFilter {

    /* renamed from: a  reason: collision with root package name */
    public static final /* synthetic */ a f4415a = new a();

    @Override // java.io.FilenameFilter
    public final boolean accept(File file, String str) {
        String str2 = CrashlyticsController.FIREBASE_CRASH_TYPE;
        return str.startsWith(CrashlyticsController.APP_EXCEPTION_MARKER_PREFIX);
    }
}