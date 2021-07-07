var script = document.createElement('script');          
script.src = "https://code.jquery.com/jquery-3.6.0.min.js";
document.head.appendChild(script)


$(document).ready(function() {
		var toc = '';
		var cntr = 0;
		var tocId;
		function getId(headerText) {
			var idStyle = 'naturalText';
			if (idStyle === 'naturalText') {
				//Change spaces to dashes, and make all lower-case. Limit to six words.
				headerText = headerText.toLocaleLowerCase();
				hdrText = headerText.split(' ', 6);
				return hdrText.join('-');
			} else {
				//the id's should be "toc1", "toc2", "toc3", etc.
				return 'toc' + ++cntr;
			}
		}
		$('h1, h2, h3').each(function(e) {
			$hdr = $(this);
			if (!(tocId = $hdr.attr('id'))) { //hdr does not already have an id
				tocId = getId($hdr.text());
				$hdr.attr('id', tocId);
			}
			var tag = $hdr[0].tagName.toLowerCase()
			var mar_left = (parseInt(tag.replaceAll("h",""))-1)*30
			var style="margin-left:"+mar_left+"px; padding-top: 5px"
			var tempbox = ""
			if (tag == 'h3'){
				style = "padding-left: 30px; border-left: black solid 1px;"
				tempbox = "<span style='width:30px; padding-top:4px; padding-bottom:2px; padding-left:30px; border-left:black solid 1px;'>&nbsp;</span>"
			}
			if (tag == 'h2'){
				style = "padding-left: 30px; padding-top:5px; border-left: black solid 1px;"
			}
			toc += '<li class="' + tag + '" style="'+style+'">'+tempbox+'<a href="#' + tocId + '" style="text-decoration:none; color:#4d94ff;">' + $hdr.text() + '</a></li>';
		});
		$('#toc').html('<ul style="list-style-type:none; padding:0px;">' + toc + '</ul>');
	});
