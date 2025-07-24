package android.support.v4.media;

import a.a.a.b.b;
import android.os.Bundle;
import android.os.Parcelable;
import android.support.v4.media.session.MediaSessionCompat;
import java.util.ArrayList;

/* loaded from: classes.dex */
public class MediaBrowserCompat$SearchResultReceiver extends b {
    @Override // a.a.a.b.b
    public void a(int i, Bundle bundle) {
        MediaSessionCompat.a(bundle);
        if (i == 0 && bundle != null && bundle.containsKey("search_results")) {
            Parcelable[] parcelableArray = bundle.getParcelableArray("search_results");
            if (parcelableArray != null) {
                ArrayList arrayList = new ArrayList();
                for (Parcelable parcelable : parcelableArray) {
                    arrayList.add((MediaBrowserCompat$MediaItem) parcelable);
                }
            }
            throw null;
        }
        throw null;
    }
}