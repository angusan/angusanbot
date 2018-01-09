<?php
	//接收request的body(可以接收除了Content-type為multipart/form-data的資料)
	$json_str = file_get_contents('php://input'); 
	$json_obj = json_decode($json_str); //轉成json格式
	
	$myfile = fopen("log.txt", "w+") or die("Unable to open file!"); //設定一個log.txt，用來印訊息
	fwrite($myfile, "\xEF\xBB\xBF".$json_str); //在字串前加上\xEF\xBB\xBF轉成utf8格式

	//產生回傳給line server的格式
	$sender_userid = $json_obj->events[0]->source->userId;
	$sender_txt = $json_obj->events[0]->message->text;
	$sender_replyToken = $json_obj->events[0]->replyToken;
	$line_server_url = 'https://api.line.me/v2/bot/message/push';
	switch ($sender_txt) {
    		case "push":
        		$response = array (
				"to" => $sender_userid,
				"messages" => array (
					array (
						"type" => "text",
						"text" => "Hello, YOU SAY ".$sender_txt
					)
				)
			);
        		break;
    		case "reply":
			$line_server_url = 'https://api.line.me/v2/bot/message/reply';
        		$response = array (
				"replyToken" => $sender_replyToken,
				"messages" => array (
					array (
						"type" => "text",
						"text" => "Hello, YOU SAY ".$sender_txt
					)
				)
			);
        		break;
		case "image":
			$line_server_url = 'https://api.line.me/v2/bot/message/reply';
        		$response = array (
				"replyToken" => $sender_replyToken,
				"messages" => array (
					array (
						"type" => "image",
						"originalContentUrl" => "https://www.w3schools.com/css/paris.jpg",
						"previewImageUrl" => "https://www.nasa.gov/sites/default/themes/NASAPortal/images/feed.png"
					)
				)
			);
        		break;
		case "location":
			$line_server_url = 'https://api.line.me/v2/bot/message/reply';
        		$response = array (
				"replyToken" => $sender_replyToken,
				"messages" => array (
					array (
						"type" => "location",
						"title" => "my location",
						"address" => "〒150-0002 東京都渋谷区渋谷２丁目２１−１",
            					"latitude" => 35.65910807942215,
						"longitude" => 139.70372892916203
					)
				)
			);
        		break;
		case "sticker":
			$line_server_url = 'https://api.line.me/v2/bot/message/reply';
        		$response = array (
				"replyToken" => $sender_replyToken,
				"messages" => array (
					array (
						"type" => "sticker",
						"packageId" => "1",
						"stickerId" => "1"
					)
				)
			);
        		break;
		case "button":
			$line_server_url = 'https://api.line.me/v2/bot/message/reply';
        		$response = array (
				"replyToken" => $sender_replyToken,
				"messages" => array (
					array (
						"type" => "template",
						"altText" => "this is a buttons template",
						"template" => array (
							"type" => "buttons",
							"thumbnailImageUrl" => "https://www.w3schools.com/css/paris.jpg",
							"title" => "Menu",
							"text" => "Please select",
							"actions" => array (
								array (
									"type" => "postback",
									"label" => "Buy",
									"data" => "action=buy&itemid=123"
								),
								array (
									"type" => "postback",
                   							"label" => "Add to cart",
                    							"data" => "action=add&itemid=123"
								)
							)
						)
					)
				)
			);
        		break;
    		default:
			$objID = $json_obj->events[0]->message->id;
			$url = 'https://api.line.me/v2/bot/message/'.$objID.'/content';
			$ch = curl_init($url);
			curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
			curl_setopt($ch, CURLOPT_HTTPHEADER, array(
				'Authorization: Bearer cCubKq3mCMRx0RcZcoHLDP0r38pPEn5ZkqgTRT0c4fexsmrtN52Fs5kGkQxZYmED5pM1iDsG5M+1si8PS5dgKDs8xF6Qw0DNdddVrMkhc9WJmD1pRVtGqwY4rSNS+/AgkfGoI10hRps8GI//6k7f9AdB04t89/1O/w1cDnyilFU=',
			));
				
			$json_content = curl_exec($ch);
			curl_close($ch);

			$imagefile = fopen($objID.".jpeg", "w+") or die("Unable to open file!"); //設定一個log.txt，用來印訊息
			fwrite($imagefile, $json_content); 
			fclose($imagefile);
        		$header[] = "Content-Type: application/json";
			$post_data = array (
				"requests" => array (
						array (
							"image" => array (
								"source" => array (
									"imageUri" => "http://139.59.123.8/chtChatBot/20180109_LineBot/".$objID.".jpeg"
								)
							),
							"features" => array (
								array (
									"type" => "TEXT_DETECTION",
									"maxResults" => 1
								)
							)
						)
					)
			);
			$ch = curl_init('https://vision.googleapis.com/v1/images:annotate?key=AIzaSyCiyGiCfjzzPR1JS8PrAxcsQWHdbycVwmg');                                                                      
			curl_setopt($ch, CURLOPT_CUSTOMREQUEST, "POST");                                                                     
			curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($post_data));                                                                  
			curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);                                                                      
			curl_setopt($ch, CURLOPT_HTTPHEADER, $header);                                                                                                   
			$result = json_decode(curl_exec($ch));
			$result_ary = mb_split("\n",$result -> responses[0] -> fullTextAnnotation -> text);
			$ans_txt = "這張發票沒用了，你又製造了一張垃圾";
			foreach ($result_ary as $val) {
				if($val == "AG-26272435"){
					$ans_txt = "恭喜您中獎啦，快分紅!!";
				}
			}
			$response = array (
				"to" => $sender_userid,
				"messages" => array (
					array (
						"type" => "text",
						"text" => $ans_txt
					)
				)
			);
        		break;
	}

	
	
	//fwrite($myfile, "\xEF\xBB\xBF".json_encode($response)); //在字串前加上\xEF\xBB\xBF轉成utf8格式
	fclose($myfile);
		
	//回傳給line server
	$header[] = "Content-Type: application/json";
	$header[] = "Authorization: Bearer OWBhh1ZaY04jmRRWONkvglSiSPBZo5EdKzj3Btn/1bcPrYFwSbSh2+Jeeo3vyyeDLiZP6ANrW0Wfe1M+UhxsZ6H8ZYYI1IZFOtRl4eGHJpep9odRsGxLOsiyQAK4MCfqlkX8ChXvkBzc11eUQhgcuAdB04t89/1O/w1cDnyilFU=";
	$ch = curl_init($line_server_url);                                                                      
	curl_setopt($ch, CURLOPT_CUSTOMREQUEST, "POST");                                                                     
	curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($response));                                                                  
	curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);                                                                      
	curl_setopt($ch, CURLOPT_HTTPHEADER, $header);                                                                                                   
	$result = curl_exec($ch);
	curl_close($ch);
?>
