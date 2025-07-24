package c.d.c.h.e.c;

import com.google.firebase.crashlytics.internal.model.CrashlyticsReport;
import java.util.Comparator;

/* compiled from: lambda */
/* loaded from: classes2.dex */
public final /* synthetic */ class d implements Comparator {

    /* renamed from: b  reason: collision with root package name */
    public static final /* synthetic */ d f4418b = new d();

    @Override // java.util.Comparator
    public final int compare(Object obj, Object obj2) {
        return ((CrashlyticsReport.CustomAttribute) obj).getKey().compareTo(((CrashlyticsReport.CustomAttribute) obj2).getKey());
    }
}