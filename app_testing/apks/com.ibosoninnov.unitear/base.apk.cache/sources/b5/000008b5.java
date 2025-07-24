package c.d.a.a.a;

import com.google.android.datatransport.cct.CctTransportBackend;
import com.google.android.datatransport.runtime.logging.Logging;
import com.google.android.datatransport.runtime.retries.RetryStrategy;
import java.net.URL;

/* compiled from: lambda */
/* loaded from: classes.dex */
public final /* synthetic */ class a implements RetryStrategy {

    /* renamed from: a  reason: collision with root package name */
    public static final /* synthetic */ a f4206a = new a();

    @Override // com.google.android.datatransport.runtime.retries.RetryStrategy
    public final Object shouldRetry(Object obj, Object obj2) {
        CctTransportBackend.HttpRequest httpRequest = (CctTransportBackend.HttpRequest) obj;
        CctTransportBackend.HttpResponse httpResponse = (CctTransportBackend.HttpResponse) obj2;
        URL url = httpResponse.redirectUrl;
        if (url != null) {
            Logging.d("CctTransportBackend", "Following redirect to: %s", url);
            return httpRequest.withUrl(httpResponse.redirectUrl);
        }
        return null;
    }
}